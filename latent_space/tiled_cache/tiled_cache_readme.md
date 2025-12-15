# Adding Tiled Cache to MLExchange File Manager

This guide explains how to add Tiled client-side caching to the MLExchange File Manager when used in the Latent Space Explorer application.

## Overview

The MLExchange File Manager connects to Tiled for data access. By implementing client-side caching, we can significantly improve performance when accessing the same data multiple times, which is common in the Latent Space Explorer application.

## Implementation Steps

### 1. Update your .env file

Add the following Tiled cache configuration variables to your `.env` file:

```
# Tiled cache configuration
TILED_CACHE_PATH=/tiled_storage/tiled_cache.db
TILED_CACHE_CAPACITY=500000000
TILED_CACHE_MAX_ITEM_SIZE=500000
```

### 2. Update docker-compose.yml

Add the Tiled cache environment variables to the `latent-space-explorer` service:

```yaml
latent-space-explorer:
  # Existing configuration...
  environment:
    # Existing environment variables...
    
    # Add these lines
    TILED_CACHE_PATH: '${TILED_CACHE_PATH:-/tiled_storage/tiled_cache.db}'
    TILED_CACHE_CAPACITY: '${TILED_CACHE_CAPACITY:-500000000}'
    TILED_CACHE_MAX_ITEM_SIZE: '${TILED_CACHE_MAX_ITEM_SIZE:-500000}'
```

### 3. Fork the File Manager Repository

Since you're using the file_manager as a dependency via git, you'll need to make the file_manager recognize these environment variables by forking the repository:

1. Fork the mlex_file_manager repository
2. Modify the `file_manager/dataset/tiled_dataset.py` file to include cache support
3. Update your dependency in pyproject.toml to point to your fork

Here's the code to add to the `get_tiled_client` method in `tiled_dataset.py`:

```python
@staticmethod
def get_tiled_client(
    tiled_uri, api_key=None, static_tiled_client=STATIC_TILED_CLIENT
):
    """
    Get the tiled client
    Args:
        tiled_uri:              Tiled URI
        api_key:                Tiled API key
        static_tiled_client:    Static tiled client
    Returns:
        Tiled client
    """
    # Checks if a static tiled client has been set, otherwise creates a new one
    if static_tiled_client:
        return static_tiled_client
    else:
        # Check for environment variables for cache configuration
        cache_path = os.environ.get("TILED_CACHE_PATH")
        
        if cache_path:
            # Import cache
            from tiled.client.cache import Cache
            
            # Get capacity and max item size from environment or use defaults
            capacity = int(os.environ.get("TILED_CACHE_CAPACITY", 500_000_000))
            max_item_size = int(os.environ.get("TILED_CACHE_MAX_ITEM_SIZE", 500_000))
            
            # Create custom cache
            cache = Cache(
                capacity=capacity,
                max_item_size=max_item_size,
                filepath=cache_path,
                readonly=False,
            )
            
            # Create client with custom cache
            client = from_uri(tiled_uri, api_key=api_key, cache=cache)
        else:
            # Create client with default cache
            client = from_uri(tiled_uri, api_key=api_key)
            
        return client
```

### 4. Update Your Dependency

After forking the repository and making the changes, update your dependency in `pyproject.toml`:

```toml
dependencies = [
    # Other dependencies...
    "mlex_file_manager@git+https://github.com/YOUR_USERNAME/mlex_file_manager.git",
    # Rest of dependencies...
]
```

Replace `YOUR_USERNAME` with your GitHub username where you've created the fork.

## Benefits of Caching

1. **Performance**: Significantly faster data access for repeated requests
2. **Reduced Network Load**: Less traffic between the application and Tiled server
3. **Better User Experience**: Quicker response times for the UI

## Monitoring and Maintenance

- Check the cache file size periodically to ensure it doesn't grow too large
- If you encounter issues with stale data, you may need to clear the cache
- The cache is stored at the path specified in `TILED_CACHE_PATH`

## Troubleshooting

If the cache doesn't seem to be working:

1. Verify that the environment variables are correctly set in your container
2. Check that the cache file path is writable
3. Look for any error messages in the logs related to Tiled or cache initialization
4. Try increasing the cache capacity if you're working with large datasets
