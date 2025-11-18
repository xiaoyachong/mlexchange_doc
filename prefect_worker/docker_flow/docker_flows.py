import tempfile
import os
import yaml
from prefect import context, flow
from prefect.states import Failed
from prefect_docker.containers import (
    create_docker_container,
    start_docker_container,
    stop_docker_container,
    remove_docker_container,
    get_docker_container_logs,
)

from flows.docker.schema import DockerParams
from flows.logger import setup_logger
from flows.credentials import add_credentials_to_io_parameters


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
        
        try:
            # Create container using prefect-docker
            # The volumes parameter in the underlying Docker SDK expects a list of strings
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
                # Pass volumes as part of **create_kwargs
                volumes=volumes,
            )
            
            container_id = container.id
            logger.info(f"Created container {container_id}")
            
            # Start container
            await start_docker_container(container_id=container_id)
            logger.info(f"Started container {container_id}")
            
            # Wait for container and stream logs
            import docker
            client = docker.from_env()
            running_container = client.containers.get(container_id)
            
            # Stream logs in real-time
            for log_line in running_container.logs(stream=True, follow=True):
                logger.info(log_line.decode('utf-8').rstrip())
            
            # Check exit code
            running_container.reload()
            exit_code = running_container.attrs['State']['ExitCode']
            
            if exit_code != 0:
                logger.error(f"Container exited with status {exit_code}")
                return Failed(message=f"Docker command failed with status {exit_code}")
            
            logger.info("Container completed successfully")
                    
        except Exception as e:
            logger.error(f"Error running container: {str(e)}")
            
            # Try to get logs if container exists
            if container_id:
                try:
                    logs = await get_docker_container_logs(container_id=container_id)
                    logger.error(f"Container logs: {logs}")
                except:
                    pass
            
            return Failed(message=f"Docker container error: {str(e)}")
        
        finally:
            # Cleanup container
            if container_id:
                try:
                    await stop_docker_container(container_id=container_id, timeout=10)
                    logger.info(f"Stopped container {container_id}")
                except Exception as stop_error:
                    logger.warning(f"Error stopping container: {stop_error}")
                
                try:
                    await remove_docker_container(container_id=container_id, force=True)
                    logger.info(f"Removed container {container_id}")
                except Exception as remove_error:
                    logger.warning(f"Error removing container: {remove_error}")
            
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass

    return current_flow_run_id