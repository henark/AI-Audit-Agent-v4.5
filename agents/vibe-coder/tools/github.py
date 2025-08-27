import os, tempfile, subprocess, time
from github import Github

GH = Github(os.getenv("ZERO_GITHUB_TOKEN"))
REPO = GH.get_repo("Aurumgrid/Z-n-")

def open_pr(title, patch, rationale):
    token = os.getenv("ZERO_GITHUB_TOKEN")
    if not token:
        raise ValueError("ZERO_GITHUB_TOKEN environment variable not set")

    # Reconfigure git remote to use the token for authentication
    repo_slug = "Aurumgrid/Z-n-"
    auth_url = f"https://x-access-token:{token}@github.com/{repo_slug}.git"
    subprocess.run(["git", "remote", "set-url", "origin", auth_url], check=True)

    branch = f"auto/{int(time.time())}"
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "apply"], input=patch.encode(), check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", f"{title}\n\n{rationale}"], check=True)
    subprocess.run(["git", "push", "origin", branch], check=True)
    pr = REPO.create_pull(title=title, body=rationale, head=branch, base="main")
    return pr.html_url
