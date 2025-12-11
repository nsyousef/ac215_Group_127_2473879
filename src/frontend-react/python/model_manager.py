"""
Model manager for downloading and caching ML models securely.

Supports multiple hosting backends (GitHub Releases, HuggingFace) with
pluggable architecture for easy switching and testing.

SETUP INSTRUCTIONS:
===================
1. Create a PUBLIC GitHub repo for model releases (separate from code repo):
   Example: https://github.com/nsyousef/ac215_pibu_ai_models
   - Make it PUBLIC (important!)
   - Add a README explaining it's for model storage only

2. Create a release in the PUBLIC model repo:
   - Tag: v0.0.1 (or your version)
   - Upload: vision_model_v0.0.1.pth

3. Copy the download link from the release and update MODEL_SOURCES below.
   Public releases = no authentication needed, no egress costs, no secrets!

4. Test locally:
   python -c "from model_manager import get_model_path; print(get_model_path('vision'))"

SECURITY:
=========
- GitHub Release URLs are PUBLIC (no credentials needed)
- Safe for production apps (no token exposure)
- Models cached locally to ~/.cache/Pibu/models/
- Optional fallback to HuggingFace or other sources

ARCHITECTURE:
=============
- MODEL_SOURCES: Priority list of where to download models from
- _download_from_source(): Routes to appropriate backend
- _download_github_release(): Download from public GitHub release (current)
- _download_huggingface(): Download from HF Hub (future migration)
- get_model_path(): Simple interface for api_manager.py

FUTURE MIGRATION:
=================
Add HF as fallback source in MODEL_SOURCES without changing any code logic.
Just update the list of sources - the rest works unchanged.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import requests


def debug_log(msg: str):
    """Print to stderr so it doesn't interfere with stdout JSON protocol"""
    print(msg, file=sys.stderr, flush=True)


# ============================================================================
# Model Sources Configuration
# ============================================================================
# Priority list of sources to try (in order) when downloading models
# UPDATE the GitHub Release URL with your actual release link
# ============================================================================

MODEL_SOURCES = {
    "vision": [
        # GitHub Release from PUBLIC model repo (no auth needed)
        # Update URL after creating public repo: github.com/nsyousef/pibu_ai_model_release
        {
            "url": "https://github.com/nsyousef/pibu_ai_model_release/releases/download/v0.0.2/vision_model_v0.0.2.pth",
            "backend": "github_release",
            "requires_auth": False,
        },
        # HuggingFace fallback (for future migration)
        {
            "url": "PLACEHOLDER_ORG/PLACEHOLDER_MODEL",  # HF repo ID format
            "filename": "vision_model_v1.pth",
            "backend": "huggingface",
            "requires_auth": False,
        },
    ],
}


def get_model_path(
    model_type: str = "vision",
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Path:
    """
    Get path to model file, downloading if necessary.

    Tries each source in MODEL_SOURCES priority order until one succeeds.

    Args:
        model_type: "vision" or other model type key in MODEL_SOURCES
        progress_callback: Optional callback(percent) for progress tracking

    Returns:
        Path to model checkpoint

    Raises:
        FileNotFoundError: If model cannot be downloaded from any source
    """
    cache_dir = Path.home() / ".cache" / "Pibu" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sources = MODEL_SOURCES.get(model_type, [])
    if not sources:
        raise ValueError(f"Unknown model type: {model_type}")

    # Try each source in priority order
    last_error = None
    for source in sources:
        try:
            return _download_from_source(
                source=source,
                model_type=model_type,
                cache_dir=cache_dir,
                progress_callback=progress_callback,
            )
        except Exception as e:
            last_error = e
            debug_log(f"WARNING: Source failed ({source['backend']}): {e}")
            continue

    # All sources failed
    if last_error:
        raise FileNotFoundError(f"Could not download {model_type} model from any source. " f"Last error: {last_error}")
    raise FileNotFoundError(f"No sources configured for {model_type} model")


def _download_from_source(
    source: Dict[str, Any],
    model_type: str,
    cache_dir: Path,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Path:
    """
    Download model from a single source.

    Args:
        source: Dict with 'backend', 'url', and optional 'requires_auth'
        model_type: "vision" or other type
        cache_dir: Directory to cache downloaded model
        progress_callback: Optional progress callback

    Returns:
        Path to downloaded model file
    """
    backend = source["backend"]

    if backend == "github_release":
        return _download_github_release(
            url=source["url"],
            model_type=model_type,
            cache_dir=cache_dir,
            progress_callback=progress_callback,
        )
    elif backend == "huggingface":
        return _download_huggingface(
            repo_id=source["url"],
            filename=source.get("filename", f"{model_type}_model.pth"),
            model_type=model_type,
            cache_dir=cache_dir,
            progress_callback=progress_callback,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _download_github_release(
    url: str,
    model_type: str,
    cache_dir: Path,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Path:
    """
    Download from public GitHub release (no authentication needed).

    GitHub Release assets are always publicly accessible.

    Args:
        url: Full download URL from GitHub release
        model_type: "vision" or other type
        cache_dir: Cache directory
        progress_callback: Optional progress callback

    Returns:
        Path to downloaded file
    """
    # Extract filename from URL
    filename = url.split("/")[-1]
    cache_path = cache_dir / filename

    # Return if already cached
    if cache_path.exists():
        debug_log(f"Using cached model: {cache_path}")
        return cache_path

    debug_log(f"Downloading {model_type} model from GitHub Release...")
    debug_log(f"    URL: {url}")

    try:
        # Add headers to ensure GitHub returns the file, not HTML
        headers = {
            "Accept": "application/octet-stream",
            "User-Agent": "pibu-ai-model-downloader",
        }
        response = requests.get(
            url,
            stream=True,
            timeout=300,
            headers=headers,
            allow_redirects=True,
        )
        debug_log(f"    HTTP Status: {response.status_code}")
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        debug_log(f"    Content-Length: {total_size} bytes")
        downloaded = 0

        with open(cache_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if progress_callback:
                            progress_callback(progress)
                        print(
                            f"    Progress: {progress:.1f}%",
                            end="\r",
                            file=sys.stderr,
                            flush=True,
                        )

        print("", file=sys.stderr)  # Newline after progress
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        debug_log(f"Model downloaded: {cache_path} ({size_mb:.1f} MB)")
        return cache_path

    except requests.exceptions.RequestException as e:
        if cache_path.exists():
            cache_path.unlink()
        debug_log(f"✗ Download failed: {e}")
        if "response" in locals():
            debug_log(f"  Response status: {response.status_code}")
            debug_log(f"  Response headers: {dict(response.headers)}")
        raise FileNotFoundError(f"Failed to download from GitHub: {e}")


def _download_huggingface(
    repo_id: str,
    filename: str,
    model_type: str,
    cache_dir: Path,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Path:
    """
    Download from Hugging Face Model Hub.

    Args:
        repo_id: HF repo ID (e.g., "ac215/pibu-ai-vision-model")
        filename: Filename within the repo
        model_type: "vision" or other type
        cache_dir: Cache directory
        progress_callback: Optional progress callback

    Returns:
        Path to downloaded file
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub not installed. " "Install with: pip install huggingface-hub")

    debug_log(f"Downloading {model_type} model from Hugging Face...")
    debug_log(f"    Repo: {repo_id}")

    try:
        # HF automatically handles caching
        hf_token = os.getenv("HF_TOKEN")

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            token=hf_token,
            resume_download=True,
        )

        debug_log(f"Model downloaded from HF: {model_path}")
        return Path(model_path)

    except Exception as e:
        raise FileNotFoundError(f"Failed to download from Hugging Face: {e}")
