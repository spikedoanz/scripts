#https://gist.github.com/spikedoanz/ef28a2464b0c781d645ee3c9465c2c82
import time
import asyncio
import asyncssh
import os
import ray
import base64
import signal
import multiprocessing
from pathlib import Path
from typing import List, Optional, Dict
import logging
import dacite

class SerialDataclass:
    """Base class for dataclasses with JSON serialization support"""
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(asdict(self), ensure_ascii=False) #type:ignore
    
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize from JSON string using dacite"""
        data = json.loads(json_str)
        return dacite.from_dict(data_class=cls, data=data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)  # type: ignore

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create instance from dictionary using dacite"""
        return dacite.from_dict(data_class=cls, data=data)
    
    def to_path(self, path: str | Path) -> None:
        """Save to file as JSON"""
        Path(path).write_text(self.to_json(), encoding="utf-8")
    
    @classmethod
    def from_path(cls: Type[T], path: str | Path) -> T:
        """Load from JSON file"""
        json_str = Path(path).read_text(encoding="utf-8")
        return cls.from_json(json_str)

    def hash(self) -> int:
        return hash(self.to_json()) 

    def __repr__(self) -> str:
        """Return pretty-formatted JSON representation"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2) #type:ignore

@dataclass(frozen=True)
class SSHConfig(SerialDataclass):
    hostname: str
    username: str
    private_key: str  # PEM-format string
    port: int = 22
    port_forwards: List[PortForward] = field(default_factory=list)
    private_key_passphrase: Optional[str] = None
    keepalive_interval: int = 30
    keepalive_count_max: int = 3
    exit_on_forward_failure: bool = True
    
    @classmethod
    def from_key_file(
        cls, 
        hostname: str,
        username: str,
        key_path: Path = DEFAULT_CLUSTER_KEY_PATH,
        **kwargs
    ) -> "SSHConfig":
        """Create SSHConfig from a key file path"""
        if not key_path.exists():
            raise FileNotFoundError(f"SSH key not found at {key_path}")
            
        return cls(
            hostname=hostname,
            username=username,
            private_key=key_path.read_text(),
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert PortForward objects to dicts
        data['port_forwards'] = [pf.to_dict() for pf in self.port_forwards]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSHConfig":
        # Convert port_forwards back to PortForward objects
        if 'port_forwards' in data:
            data['port_forwards'] = [PortForward.from_dict(pf) for pf in data['port_forwards']]
        return cls(**data)

@dataclass(frozen=True)
class PortForward(SerialDataclass):
    local_port: int
    remote_host: str
    remote_port: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortForward":
        return cls(**data)

logger = logging.getLogger(__name__)

def verbose(level=1):
    # verbose(1): short, concise info
    # verbose(2): diagnostics, logs, errors
    # verbose(3): full logs
    # verbose(4): sanity checks.
    return int(os.getenv("VERBOSE", 0)) >= level

def free_port() -> int:
    """Ask the kernel for an unused local TCP port, then release it."""
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

class ManagedSSHAgent:
    """In-memory SSH agent that never touches the filesystem"""

    def __init__(self):
        self._keys: Dict[str, asyncssh.SSHKey] = {}

    async def add_key(self, key_data: str, passphrase: Optional[str] = None):
        key = asyncssh.import_private_key(key_data, passphrase)
        fingerprint = key.get_fingerprint()
        self._keys[fingerprint] = key
        return fingerprint

    def get_keys(self):
        return list(self._keys.values())

async def ssh_run(ssh_config: SSHConfig, command: str) -> Optional[str]:
    agent = ManagedSSHAgent()
    await agent.add_key(ssh_config.private_key, ssh_config.private_key_passphrase)
    try:
        async with asyncssh.connect(
            host=ssh_config.hostname,
            port=ssh_config.port,  # Use port from config
            username=ssh_config.username,
            client_keys=agent.get_keys(),
            known_hosts=None,
        ) as conn:
            result = await conn.run(command)
            if verbose(4): 
                print(result)
            return result.stdout if result.exit_status == 0 else None  # type: ignore
    except Exception as e:  # catches indexing errors from result.stdout
        logger.warning(f"SSH command '{command}' failed: {e}")
        return None

class SSHTunnelManager:
    """Manages SSH connections with port forwarding and lifecycle management"""
    
    def __init__(self, config: SSHConfig, parent_coupled: bool = True):
        self.config = config
        self.parent_coupled = parent_coupled
        self._connection: Optional[asyncssh.SSHClientConnection] = None
        self._listeners: List[asyncssh.SSHListener] = []
        self._agent = ManagedSSHAgent()
        self._shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        
        # Weak reference to track parent process
        if parent_coupled:
            self._setup_parent_monitoring()
    
    def _setup_parent_monitoring(self):
        """Monitor parent process and shutdown if it dies"""
        parent_pid = os.getppid()
        
        async def monitor_parent():
            while not self._shutdown_event.is_set():
                try:
                    # Check if parent is still alive
                    os.kill(parent_pid, 0)
                except ProcessLookupError:
                    logger.warning("Parent process died, initiating shutdown")
                    await self.shutdown()
                    break
                await asyncio.sleep(1)
        
        task = asyncio.create_task(monitor_parent())
        self._tasks.append(task)
    
    async def _create_tunnel(self, forward: PortForward) -> asyncssh.SSHListener:
        """Create a single port forward tunnel"""
        try:
            assert self._connection, "No ssh connection established"
            listener = await self._connection.forward_local_port(
                listen_host='127.0.0.1',
                listen_port=forward.local_port,
                dest_host=forward.remote_host,
                dest_port=forward.remote_port
            )
            logger.info(f"Established tunnel: localhost:{forward.local_port} -> "
                       f"{forward.remote_host}:{forward.remote_port}")
            return listener
        except Exception as e:
            logger.error(f"Failed to create tunnel for port {forward.local_port}: {e}")
            if self.config.exit_on_forward_failure:
                raise e
            else:
                raise e
    
    async def connect(self):
        """Establish SSH connection and set up port forwards"""
        # Add private key to our in-memory agent
        await self._agent.add_key(self.config.private_key, self.config.private_key_passphrase)
        
        # Connection options
        options = {
            'host': self.config.hostname,
            'username': self.config.username,
            'client_keys': self._agent.get_keys(),
            'known_hosts': None,  # You might want to handle this properly
            'keepalive_interval': self.config.keepalive_interval,
            'keepalive_count_max': self.config.keepalive_count_max,
        }
        
        try:
            self._connection = await asyncssh.connect(**options)
            logger.info(f"Connected to {self.config.username}@{self.config.hostname}")
            
            # Set up port forwards
            for forward in self.config.port_forwards:
                listener = await self._create_tunnel(forward)
                if listener:
                    self._listeners.append(listener)
            
            # Set up signal handlers for graceful shutdown
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            await self.shutdown()
            raise
    
    async def wait(self):
        """Wait until shutdown is triggered"""
        await self._shutdown_event.wait()
    
    async def shutdown(self):
        """Gracefully shutdown all connections and listeners"""
        logger.info("Initiating shutdown...")
        self._shutdown_event.set()
        
        # Close all port forward listeners
        for listener in self._listeners:
            listener.close()
            await listener.wait_closed()
        
        # Close SSH connection
        if self._connection:
            self._connection.close()
            await self._connection.wait_closed()
        
        # Cancel monitoring tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        logger.info("Shutdown complete")

class SSHTunnelProcess:
    """Simple SSH tunnel that runs in a subprocess"""
    
    def __init__(self, config: SSHConfig):
        self.config = config
        self._process: Optional[multiprocessing.Process] = None
        self._ready_event = multiprocessing.Event()
        self._error_queue = multiprocessing.Queue()
    
    def start(self, timeout: float = 30.0) -> bool:
        """This is blocking"""
        if self._process and self._process.is_alive():
            logger.warning("Tunnel already running")
            return False
        
        # Reset events
        self._ready_event.clear()
        
        self._process = multiprocessing.Process(
            target=self._run_tunnel,
            args=(self.config, self._ready_event, self._error_queue)
        )
        self._process.daemon = True
        self._process.start()
        
        # Wait for tunnel to be ready or timeout
        if self._ready_event.wait(timeout=timeout):
            logger.info(f"SSH tunnel established with PID {self._process.pid}")
            return True
        else:
            # Check if process died
            if not self._process.is_alive():
                # Try to get error message
                try:
                    error = self._error_queue.get_nowait()
                    logger.error(f"SSH tunnel failed: {error}")
                except:
                    logger.error("SSH tunnel process died without establishing connection")
            else:
                logger.error("SSH tunnel timed out during startup")
            
            # Clean up
            self.stop()
            return False
    
    def stop(self, timeout: float = 5.0) -> bool:
        """Stop the tunnel"""
        if not self._process or not self._process.is_alive():
            return True
        
        self._process.terminate()
        self._process.join(timeout=timeout)
        
        if self._process.is_alive():
            logger.warning("Process didn't terminate, killing")
            self._process.kill()
            self._process.join()
        
        # Clean up queue to prevent resource leaks
        while not self._error_queue.empty():
            try:
                self._error_queue.get_nowait()
            except:
                break
        
        return True
        
    def is_running(self) -> bool:
        """Check if tunnel is running"""
        return self._process is not None and self._process.is_alive()
    
    @staticmethod
    def _run_tunnel(config: SSHConfig, ready_event: multiprocessing.Event, #type:ignore
                    error_queue: multiprocessing.Queue):
        """Run the tunnel in the subprocess"""
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        async def run():
            try:
                # Import private key from PEM string
                pkey = asyncssh.import_private_key(
                    config.private_key,
                    passphrase=config.private_key_passphrase
                )
                
                # Create connection options
                connect_kwargs = {
                    'host': config.hostname,
                    'port': config.port,
                    'username': config.username,
                    'client_keys': [pkey],
                    'known_hosts': None,  # Disable host key checking
                    'keepalive_interval': config.keepalive_interval,
                    'keepalive_count_max': config.keepalive_count_max,
                }
                
                # If we have port forwards, set up the listeners
                listeners = []
                
                async with asyncssh.connect(**connect_kwargs) as conn:
                    logger.info(f"SSH connection established to {config.username}@{config.hostname}:{config.port}")
                    
                    # Set up port forwards
                    for pf in config.port_forwards:
                        try:
                            listener = await conn.forward_local_port(
                                '127.0.0.1', pf.local_port,
                                pf.remote_host, pf.remote_port
                            )
                            listeners.append(listener)
                            logger.info(f"Port forward established: localhost:{pf.local_port} -> {pf.remote_host}:{pf.remote_port}")
                        except Exception as e:
                            error_msg = f"Failed to set up port forward {pf.local_port}->{pf.remote_host}:{pf.remote_port}: {e}"
                            logger.error(error_msg)
                            if config.exit_on_forward_failure:
                                raise RuntimeError(error_msg)
                    
                    # Signal that we're ready
                    ready_event.set()
                    
                    # Keep the connection alive
                    try:
                        # Wait indefinitely (until process is terminated)
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        logger.info("SSH tunnel cancelled")
                    finally:
                        # Clean up listeners
                        for listener in listeners:
                            listener.close()
                            await listener.wait_closed()
                        logger.info("Port forwards closed")
                        
            except Exception as e:
                error_msg = f"SSH tunnel error: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                error_queue.put(error_msg)
                raise
        
        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            logger.info("SSH tunnel interrupted")
        except Exception as e:
            logger.error(f"SSH tunnel crashed: {e}")

async def ssh_ok(
    ssh_config: SSHConfig, 
    max_wait: int = 300, 
    check_interval: int = 5
) -> bool:
    start_time = time.time()
    attempt = 0
    
    while (time.time() - start_time) < max_wait:
        attempt += 1
        logger.info((
            f"SSH connection attempt {attempt} "
            f"to {ssh_config.hostname}:{ssh_config.port}..."
        ))
        
        result = await ssh_run(ssh_config, "echo 'SSH OK'")
        if result is not None and "SSH OK" in result:
            logger.info("SSH connection established!")
            return True
            
        if (time.time() - start_time) < max_wait:
            await asyncio.sleep(check_interval)
    
    logger.error((
        f"SSH connection timeout after {max_wait} seconds"))
    return False

async def ssh_copy(ssh_config: SSHConfig, local: Path, remote: str) -> None:
    """Upload file using asyncssh's SFTP (works with in-memory keys)."""
    
    local = local.resolve()
    
    # Use the same SSH agent approach as discovery
    agent = ManagedSSHAgent()
    await agent.add_key(ssh_config.private_key, ssh_config.private_key_passphrase)
    
    async with asyncssh.connect(
        host=ssh_config.hostname,
        port=ssh_config.port,
        username=ssh_config.username,
        client_keys=agent.get_keys(),
        known_hosts=None
    ) as conn:
        async with conn.start_sftp_client() as sftp:
            # Ensure remote directory exists
            remote_dir = os.path.dirname(remote)
            try:
                await sftp.makedirs(remote_dir)
            except:
                pass  # Directory might already exist
            
            # Upload file
            await sftp.put(str(local), remote)
            print(f"[UPLOAD] {local} -> {remote} (via SFTP)")

def setup_ray_tunnel(hostname="170.9.234.79", username="ubuntu", verbose=False):
    # points to lambda cluster. suspected to be deprecated
    _free_port = free_port()
    ray_tunnel_config = SSHConfig(
        hostname=hostname,
        username=username,
        private_key=base64.b64decode(os.environ['RAY_SSH_KEY']).decode('utf-8'),
        private_key_passphrase=os.environ.get("PASSPHRASE"),
        port_forwards=[
            PortForward(_free_port,  'localhost', 10001),
        ],
    )
    ray_tunnel = SSHTunnelProcess(ray_tunnel_config)
    assert ray_tunnel.start()
    ray.init(address=f"ray://localhost:{free_port}", log_to_driver=verbose)
    return ray_tunnel
