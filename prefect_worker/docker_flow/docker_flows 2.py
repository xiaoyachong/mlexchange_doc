import tempfile
import os
import yaml
import signal
import docker
from prefect import context, flow, task
from prefect.states import Failed
from prefect_docker.containers import (
    create_docker_container,
    start_docker_container,
    stop_docker_container,
    remove_docker_container,
)

from flows.docker.schema import DockerParams
from flows.logger import setup_logger
from flows.credentials import add_credentials_to_io_parameters


@task
async def run_container_with_logs(container_id: str, logger):
    """Run container and stream logs - will be cancelled if flow is cancelled"""
    client = docker.from_env()
    
    try:
        running_container = client.containers.get(container_id)
        logger.info(f"Streaming logs from container {container_id[:12]}")
        
        # Stream logs
        for log_line in running_container.logs(stream=True, follow=True):
            log_text = log_line.decode('utf-8').rstrip()
            if log_text:
                logger.info(log_text)
        
        # Check exit code
        running_container.reload()
        exit_code = running_container.attrs['State']['ExitCode']
        return exit_code
        
    except Exception as e:
        logger.error(f"Error streaming logs: {str(e)}")
        raise


@flow(name="Docker flow")
async def launch_docker(
    docker_params: DockerParams,
    prev_flow_run_id: str = "",
):
    logger = setup_logger()

    if (
        prev_flow_run_id != ""
        and docker_params.params["io_parameters"]["uid_retrieve"] == ""
    ):
        docker_params.params["io_parameters"]["uid_retrieve"] = prev_flow_run_id

    current_flow_run_id = str(context.get_run_context().flow_run.id)
    docker_params.params["io_parameters"]["uid_save"] = current_flow_run_id

    # Add credentials to io_parameters at the child flow level
    docker_params.params = add_credentials_to_io_parameters(docker_params.params)

    # Get paths from environment variables
    container_work_dir = os.getenv("CONTAINER_WORK_DIR", "/mlex_prefect_worker")
    host_work_dir = os.getenv("PREFECT_WORK_DIR", os.getcwd())
    
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(container_work_dir, "tmp")
    os.makedirs(temp_dir, exist_ok=True)

    # Create temporary file for parameters in the mounted directory
    with tempfile.NamedTemporaryFile(mode="w+t", dir=temp_dir, delete=False) as temp_file:
        yaml.dump(docker_params.params, temp_file)
        temp_file.flush()
        
        logger.info(f"Parameters file: {temp_file.name}")
        
        # Convert container path to host path for Docker volume mounting
        host_temp_path = temp_file.name.replace(container_work_dir, host_work_dir)

        # Prepare volumes list
        volumes = docker_params.volumes + [
            f"{host_temp_path}:/app/work/config/params.yaml"
        ]
        
        # Build command
        command = f"{docker_params.command} /app/work/config/params.yaml"
        
        container = None
        container_id = None
        
        # Define signal handler inside flow to capture container_id
        def cleanup_on_signal(signum, frame):
            """Signal handler to cleanup container on SIGTERM/SIGINT"""
            if container_id:
                logger.info(f"Received signal {signum}, cleaning up container {container_id[:12]}")
                try:
                    client = docker.from_env()
                    c = client.containers.get(container_id)
                    c.stop(timeout=5)
                    c.remove(force=True)
                    logger.info(f"Container {container_id[:12]} cleaned up successfully")
                except Exception as e:
                    logger.error(f"Error in signal handler: {e}")
        
        # Register signal handlers (only in main thread, will fail silently if not)
        try:
            signal.signal(signal.SIGTERM, cleanup_on_signal)
            signal.signal(signal.SIGINT, cleanup_on_signal)
            logger.info("Signal handlers registered")
        except ValueError as e:
            logger.warning(f"Could not register signal handlers (not in main thread): {e}")
        
        try:
            # Create container using prefect-docker
            container = await create_docker_container(
                image=f"{docker_params.image_name}:{docker_params.image_tag}",
                command=["/bin/sh", "-c", command],
                environment=docker_params.env_vars,
                detach=False,
                network=docker_params.network if docker_params.network else None,
                labels={
                    "prefect.flow_run_id": current_flow_run_id,
                    "managed_by": "prefect"
                },
                volumes=volumes,
            )
            
            container_id = container.id
            logger.info(f"Created container {container_id[:12]}")
            
            # Start container
            await start_docker_container(container_id=container_id)
            logger.info(f"Started container {container_id[:12]}")
            
            # Run and stream logs - this will be interrupted on cancellation
            exit_code = await run_container_with_logs(container_id, logger)
            
            if exit_code != 0:
                logger.error(f"Container exited with status {exit_code}")
                return Failed(message=f"Docker command failed with status {exit_code}")
            
            logger.info("Container completed successfully")
                    
        except Exception as e:
            logger.error(f"Error running container: {str(e)}")
            return Failed(message=f"Docker container error: {str(e)}")
        
        finally:
            # CRITICAL: This cleanup MUST run on cancellation
            if container_id:
                logger.info(f"Cleaning up container {container_id[:12]}")
                
                # Stop container - use Docker SDK for reliability
                try:
                    client = docker.from_env()
                    c = client.containers.get(container_id)
                    
                    if c.status == 'running':
                        logger.info(f"Stopping running container {container_id[:12]}")
                        c.stop(timeout=5)
                    
                    logger.info(f"Removing container {container_id[:12]}")
                    c.remove(force=True)
                    logger.info(f"Successfully cleaned up container {container_id[:12]}")
                    
                except docker.errors.NotFound:
                    logger.info(f"Container {container_id[:12]} already removed")
                except Exception as cleanup_error:
                    logger.error(f"Cleanup error: {cleanup_error}")
                    # Last resort - force remove
                    try:
                        await stop_docker_container(container_id=container_id, timeout=5)
                        await remove_docker_container(container_id=container_id, force=True)
                    except:
                        pass
            
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass

    return current_flow_run_id