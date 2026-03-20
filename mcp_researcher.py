"""
MCP Ecosystem Researcher - Phase 1 of the autoresearch pipeline.
Crawls GitHub, mcpservers.org, npm, PyPI, Reddit, and other sources
to build a structured dataset + training corpus for MCP newsletter generation.

Usage: python3 mcp_researcher.py
"""

import os
import re
import csv
import json
import time
import logging
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results")
CORPUS_DIR = Path("corpus")
GITHUB_API = "https://api.github.com"
NPM_REGISTRY = "https://registry.npmjs.org"
PYPI_API = "https://pypi.org/pypi"
USER_AGENT = "MCPResearcher/1.0 (newsletter research bot)"

# Rate limiting
GITHUB_DELAY = 2.0  # seconds between GitHub API calls (unauthenticated: 60/hr)
GENERAL_DELAY = 1.0  # seconds between other requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

# Use GitHub token if available (raises rate limit from 60 to 5000/hr)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
if GITHUB_TOKEN:
    session.headers.update({"Authorization": f"token {GITHUB_TOKEN}"})
    log.info("Using GitHub token for higher rate limits")
else:
    log.warning("No GITHUB_TOKEN set - GitHub API limited to 60 requests/hour")


def github_get(endpoint, params=None):
    """GET from GitHub API with rate limiting."""
    url = f"{GITHUB_API}/{endpoint.lstrip('/')}"
    time.sleep(GITHUB_DELAY)
    try:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset - time.time(), 60)
            log.warning(f"GitHub rate limited, waiting {wait:.0f}s")
            time.sleep(wait)
            resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        log.error(f"GitHub API error: {e}")
        return None


def web_get(url, delay=GENERAL_DELAY):
    """GET a web page with rate limiting."""
    time.sleep(delay)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        log.error(f"Web request error for {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------

def collect_official_repos():
    """Collect repos from the official modelcontextprotocol GitHub org."""
    log.info("Collecting official MCP repos...")
    repos = []
    page = 1
    while True:
        data = github_get("orgs/modelcontextprotocol/repos", params={
            "per_page": 100, "page": page, "sort": "updated"
        })
        if not data:
            break
        repos.extend(data)
        if len(data) < 100:
            break
        page += 1
    log.info(f"Found {len(repos)} official MCP repos")
    return repos


def collect_mcp_server_repos(max_pages=10):
    """Search GitHub for repos tagged with mcp-server topic."""
    log.info("Searching GitHub for mcp-server repos...")
    repos = []
    for page in range(1, max_pages + 1):
        data = github_get("search/repositories", params={
            "q": "topic:mcp-server",
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": page,
        })
        if not data or "items" not in data:
            break
        repos.extend(data["items"])
        if len(data["items"]) < 100:
            break
    log.info(f"Found {len(repos)} mcp-server repos on GitHub")
    return repos


def collect_mcp_search_repos(max_pages=5):
    """Search GitHub for repos mentioning 'model context protocol'."""
    log.info("Searching GitHub for 'model context protocol' repos...")
    repos = []
    for page in range(1, max_pages + 1):
        data = github_get("search/repositories", params={
            "q": "model context protocol",
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": page,
        })
        if not data or "items" not in data:
            break
        repos.extend(data["items"])
        if len(data["items"]) < 100:
            break
    log.info(f"Found {len(repos)} 'model context protocol' repos")
    return repos


def get_repo_readme(owner, repo):
    """Fetch README content for a repo."""
    data = github_get(f"repos/{owner}/{repo}/readme")
    if data and "content" in data:
        import base64
        try:
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return ""


def collect_npm_packages(max_results=200):
    """Search npm for MCP-related packages."""
    log.info("Searching npm for MCP packages...")
    packages = []
    try:
        # npm search API
        url = f"{NPM_REGISTRY}/-/v1/search"
        for query in ["mcp-server", "model-context-protocol", "@modelcontextprotocol"]:
            time.sleep(GENERAL_DELAY)
            resp = session.get(url, params={
                "text": query, "size": 100
            }, timeout=30)
            if resp.ok:
                data = resp.json()
                for obj in data.get("objects", []):
                    pkg = obj.get("package", {})
                    packages.append({
                        "name": pkg.get("name", ""),
                        "version": pkg.get("version", ""),
                        "description": pkg.get("description", ""),
                        "author": pkg.get("author", {}).get("name", "") if isinstance(pkg.get("author"), dict) else str(pkg.get("author", "")),
                        "keywords": pkg.get("keywords", []),
                        "links": pkg.get("links", {}),
                    })
    except Exception as e:
        log.error(f"npm search error: {e}")
    # Deduplicate by name
    seen = set()
    unique = []
    for p in packages:
        if p["name"] not in seen:
            seen.add(p["name"])
            unique.append(p)
    log.info(f"Found {len(unique)} npm MCP packages")
    return unique


def collect_pypi_packages():
    """Search PyPI for MCP-related packages."""
    log.info("Searching PyPI for MCP packages...")
    packages = []
    try:
        # PyPI doesn't have a search API, use simple search via web
        for query in ["mcp-server", "model-context-protocol", "mcp-client"]:
            time.sleep(GENERAL_DELAY)
            resp = session.get(f"https://pypi.org/search/", params={
                "q": query, "page": 1
            }, timeout=30)
            if not resp.ok:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for result in soup.select(".package-snippet"):
                name_el = result.select_one(".package-snippet__name")
                ver_el = result.select_one(".package-snippet__version")
                desc_el = result.select_one(".package-snippet__description")
                if name_el:
                    packages.append({
                        "name": name_el.get_text(strip=True),
                        "version": ver_el.get_text(strip=True) if ver_el else "",
                        "description": desc_el.get_text(strip=True) if desc_el else "",
                    })
    except Exception as e:
        log.error(f"PyPI search error: {e}")
    seen = set()
    unique = []
    for p in packages:
        if p["name"] not in seen:
            seen.add(p["name"])
            unique.append(p)
    log.info(f"Found {len(unique)} PyPI MCP packages")
    return unique


def collect_mcpservers_org():
    """Scrape mcpservers.org for server directory."""
    log.info("Scraping mcpservers.org...")
    servers = []
    try:
        resp = web_get("https://mcpservers.org")
        if not resp:
            return servers
        soup = BeautifulSoup(resp.text, "html.parser")
        # Extract server cards/links - structure may vary
        for link in soup.select("a[href*='/server/'], a[href*='/servers/']"):
            href = link.get("href", "")
            name = link.get_text(strip=True)
            if name and href:
                servers.append({
                    "name": name,
                    "url": urllib.parse.urljoin("https://mcpservers.org", href),
                })
        # Also try to get category pages
        for link in soup.select("a[href*='/category/'], a[href*='/categories/']"):
            href = link.get("href", "")
            cat_url = urllib.parse.urljoin("https://mcpservers.org", href)
            cat_resp = web_get(cat_url)
            if cat_resp:
                cat_soup = BeautifulSoup(cat_resp.text, "html.parser")
                for slink in cat_soup.select("a[href*='/server/'], a[href*='/servers/']"):
                    shref = slink.get("href", "")
                    sname = slink.get_text(strip=True)
                    if sname and shref:
                        servers.append({
                            "name": sname,
                            "url": urllib.parse.urljoin("https://mcpservers.org", shref),
                        })
    except Exception as e:
        log.error(f"mcpservers.org scrape error: {e}")
    seen = set()
    unique = []
    for s in servers:
        if s["name"] not in seen:
            seen.add(s["name"])
            unique.append(s)
    log.info(f"Found {len(unique)} servers on mcpservers.org")
    return unique


def collect_mcp_spec():
    """Fetch MCP spec content from spec.modelcontextprotocol.io."""
    log.info("Fetching MCP spec...")
    spec_pages = []
    try:
        resp = web_get("https://spec.modelcontextprotocol.io")
        if not resp:
            return spec_pages
        soup = BeautifulSoup(resp.text, "html.parser")
        # Get main spec text
        main_text = soup.get_text(separator="\n", strip=True)
        spec_pages.append({
            "url": "https://spec.modelcontextprotocol.io",
            "title": "MCP Specification (main)",
            "content": main_text[:50000],  # cap at 50k chars
        })
        # Follow internal links for sub-pages
        for link in soup.select("a[href]"):
            href = link.get("href", "")
            if href.startswith("/") and not href.startswith("//"):
                page_url = urllib.parse.urljoin("https://spec.modelcontextprotocol.io", href)
                page_resp = web_get(page_url)
                if page_resp:
                    page_soup = BeautifulSoup(page_resp.text, "html.parser")
                    page_text = page_soup.get_text(separator="\n", strip=True)
                    spec_pages.append({
                        "url": page_url,
                        "title": link.get_text(strip=True),
                        "content": page_text[:50000],
                    })
    except Exception as e:
        log.error(f"MCP spec fetch error: {e}")
    log.info(f"Fetched {len(spec_pages)} spec pages")
    return spec_pages


def collect_reddit_posts(subreddit, query="MCP", max_posts=50):
    """Collect Reddit posts about MCP from a subreddit using old.reddit.com."""
    log.info(f"Collecting Reddit posts from r/{subreddit}...")
    posts = []
    try:
        # Use old Reddit JSON API (no auth needed)
        url = f"https://old.reddit.com/r/{subreddit}/search.json"
        resp = session.get(url, params={
            "q": query, "sort": "new", "limit": min(max_posts, 100),
            "restrict_sr": "on", "t": "month"
        }, headers={"User-Agent": USER_AGENT}, timeout=30)
        time.sleep(2)  # Reddit rate limit
        if resp.ok:
            data = resp.json()
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                posts.append({
                    "title": post.get("title", ""),
                    "selftext": post.get("selftext", "")[:2000],
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "score": post.get("score", 0),
                    "author": post.get("author", ""),
                    "created": datetime.fromtimestamp(post.get("created_utc", 0)).isoformat(),
                    "num_comments": post.get("num_comments", 0),
                })
    except Exception as e:
        log.error(f"Reddit error for r/{subreddit}: {e}")
    log.info(f"Found {len(posts)} posts from r/{subreddit}")
    return posts


def collect_anthropic_blog():
    """Scrape Anthropic's blog/news for MCP-related announcements."""
    log.info("Checking Anthropic blog for MCP announcements...")
    articles = []
    try:
        resp = web_get("https://www.anthropic.com/news")
        if not resp:
            return articles
        soup = BeautifulSoup(resp.text, "html.parser")
        for link in soup.select("a[href*='/news/']"):
            title = link.get_text(strip=True)
            href = link.get("href", "")
            if title and ("mcp" in title.lower() or "model context" in title.lower() or "tool" in title.lower()):
                full_url = urllib.parse.urljoin("https://www.anthropic.com", href)
                articles.append({"title": title, "url": full_url})
    except Exception as e:
        log.error(f"Anthropic blog error: {e}")
    log.info(f"Found {len(articles)} MCP-related Anthropic blog posts")
    return articles


# ---------------------------------------------------------------------------
# Data processing: extract companies, contacts, servers
# ---------------------------------------------------------------------------

def extract_company_from_repo(repo):
    """Extract company/org info from a GitHub repo."""
    owner = repo.get("owner", {})
    return {
        "name": owner.get("login", ""),
        "type": owner.get("type", ""),  # User or Organization
        "url": owner.get("html_url", ""),
        "avatar": owner.get("avatar_url", ""),
    }


def process_all_repos(official_repos, topic_repos, search_repos):
    """Deduplicate and process all collected repos into servers and companies."""
    seen_ids = set()
    all_repos = []
    for repo_list in [official_repos, topic_repos, search_repos]:
        for repo in repo_list:
            repo_id = repo.get("id") or repo.get("full_name", "")
            if repo_id and repo_id not in seen_ids:
                seen_ids.add(repo_id)
                all_repos.append(repo)

    servers = []
    companies = {}
    contacts = {}

    for repo in all_repos:
        full_name = repo.get("full_name", "")
        owner_login = repo.get("owner", {}).get("login", "")
        owner_type = repo.get("owner", {}).get("type", "")

        servers.append({
            "name": repo.get("name", ""),
            "full_name": full_name,
            "description": repo.get("description", "") or "",
            "url": repo.get("html_url", ""),
            "stars": repo.get("stargazers_count", 0),
            "forks": repo.get("forks_count", 0),
            "language": repo.get("language", "") or "",
            "topics": ",".join(repo.get("topics", [])),
            "updated": repo.get("updated_at", ""),
            "owner": owner_login,
            "owner_type": owner_type,
            "license": (repo.get("license") or {}).get("spdx_id", ""),
        })

        # Track companies (organizations)
        if owner_login not in companies:
            companies[owner_login] = {
                "name": owner_login,
                "type": owner_type,
                "url": repo.get("owner", {}).get("html_url", ""),
                "servers": [],
                "total_stars": 0,
            }
        companies[owner_login]["servers"].append(repo.get("name", ""))
        companies[owner_login]["total_stars"] += repo.get("stargazers_count", 0)

        # Track contacts (repo owners)
        if owner_login not in contacts:
            contacts[owner_login] = {
                "name": owner_login,
                "type": owner_type,
                "github": repo.get("owner", {}).get("html_url", ""),
                "repos": [],
            }
        contacts[owner_login]["repos"].append(repo.get("name", ""))

    return servers, companies, contacts


# ---------------------------------------------------------------------------
# Sponsorship analysis
# ---------------------------------------------------------------------------

def analyze_sponsorship_targets(companies, servers):
    """Rank companies as sponsorship targets based on activity and reach."""
    targets = []
    for name, company in companies.items():
        if company["type"] != "Organization":
            continue
        num_servers = len(company["servers"])
        total_stars = company["total_stars"]
        # Simple scoring: stars + 100 * num_servers
        score = total_stars + 100 * num_servers
        targets.append({
            "company": name,
            "url": company["url"],
            "num_servers": num_servers,
            "total_stars": total_stars,
            "score": score,
            "servers": ", ".join(company["servers"][:5]),
        })
    targets.sort(key=lambda x: x["score"], reverse=True)
    return targets


# ---------------------------------------------------------------------------
# Corpus generation (for training)
# ---------------------------------------------------------------------------

def generate_corpus(servers, companies, contacts, npm_packages, pypi_packages,
                    spec_pages, reddit_posts, blog_articles, mcpservers_list):
    """Convert all collected data into natural language training corpus."""
    os.makedirs(CORPUS_DIR, exist_ok=True)
    file_count = 0

    # 1. Server descriptions
    log.info("Generating server corpus...")
    with open(CORPUS_DIR / "servers.txt", "w") as f:
        for s in servers:
            f.write(f"MCP Server: {s['name']}\n")
            f.write(f"Repository: {s['full_name']}\n")
            if s['description']:
                f.write(f"Description: {s['description']}\n")
            f.write(f"Stars: {s['stars']}, Forks: {s['forks']}\n")
            if s['language']:
                f.write(f"Language: {s['language']}\n")
            if s['topics']:
                f.write(f"Topics: {s['topics']}\n")
            if s['license']:
                f.write(f"License: {s['license']}\n")
            f.write(f"Owner: {s['owner']} ({s['owner_type']})\n")
            f.write(f"URL: {s['url']}\n")
            f.write("\n")
    file_count += 1

    # 2. Company profiles
    log.info("Generating company corpus...")
    with open(CORPUS_DIR / "companies.txt", "w") as f:
        for name, c in sorted(companies.items(), key=lambda x: x[1]["total_stars"], reverse=True):
            f.write(f"Company/Organization: {name}\n")
            f.write(f"Type: {c['type']}\n")
            f.write(f"GitHub: {c['url']}\n")
            f.write(f"MCP Servers: {', '.join(c['servers'])}\n")
            f.write(f"Total Stars: {c['total_stars']}\n")
            f.write("\n")
    file_count += 1

    # 3. npm packages
    log.info("Generating npm corpus...")
    with open(CORPUS_DIR / "npm_packages.txt", "w") as f:
        for p in npm_packages:
            f.write(f"npm Package: {p['name']} v{p['version']}\n")
            if p['description']:
                f.write(f"Description: {p['description']}\n")
            if p['author']:
                f.write(f"Author: {p['author']}\n")
            if p.get('keywords'):
                f.write(f"Keywords: {', '.join(p['keywords'])}\n")
            f.write("\n")
    file_count += 1

    # 4. PyPI packages
    log.info("Generating PyPI corpus...")
    with open(CORPUS_DIR / "pypi_packages.txt", "w") as f:
        for p in pypi_packages:
            f.write(f"PyPI Package: {p['name']} v{p['version']}\n")
            if p['description']:
                f.write(f"Description: {p['description']}\n")
            f.write("\n")
    file_count += 1

    # 5. MCP Spec
    log.info("Generating spec corpus...")
    with open(CORPUS_DIR / "mcp_spec.txt", "w") as f:
        for page in spec_pages:
            f.write(f"=== {page['title']} ===\n")
            f.write(f"URL: {page['url']}\n\n")
            f.write(page['content'])
            f.write("\n\n")
    file_count += 1

    # 6. Reddit posts
    log.info("Generating Reddit corpus...")
    with open(CORPUS_DIR / "reddit_posts.txt", "w") as f:
        for post in reddit_posts:
            f.write(f"Reddit Post: {post['title']}\n")
            f.write(f"Author: {post['author']} | Score: {post['score']} | Comments: {post['num_comments']}\n")
            f.write(f"Date: {post['created']}\n")
            if post['selftext']:
                f.write(f"Content: {post['selftext']}\n")
            f.write(f"URL: {post['url']}\n")
            f.write("\n")
    file_count += 1

    # 7. Blog articles
    log.info("Generating blog corpus...")
    with open(CORPUS_DIR / "blog_articles.txt", "w") as f:
        for article in blog_articles:
            f.write(f"Anthropic Blog: {article['title']}\n")
            f.write(f"URL: {article['url']}\n")
            f.write("\n")
    file_count += 1

    # 8. mcpservers.org directory
    log.info("Generating mcpservers.org corpus...")
    with open(CORPUS_DIR / "mcpservers_directory.txt", "w") as f:
        for s in mcpservers_list:
            f.write(f"MCP Server Directory Entry: {s['name']}\n")
            f.write(f"URL: {s['url']}\n")
            f.write("\n")
    file_count += 1

    # 9. Q&A pairs for training
    log.info("Generating Q&A corpus...")
    with open(CORPUS_DIR / "qa_pairs.txt", "w") as f:
        # Server Q&A
        for s in servers[:100]:  # top 100 servers
            f.write(f"Q: What is the {s['name']} MCP server?\n")
            desc = s['description'] if s['description'] else f"an MCP server by {s['owner']}"
            f.write(f"A: {s['name']} is {desc}. It has {s['stars']} stars on GitHub and is written in {s['language'] or 'an unspecified language'}. You can find it at {s['url']}.\n\n")

            if s['topics']:
                f.write(f"Q: What categories does {s['name']} belong to?\n")
                f.write(f"A: {s['name']} is tagged with the following topics: {s['topics']}.\n\n")

        # Company Q&A
        for name, c in sorted(companies.items(), key=lambda x: x[1]["total_stars"], reverse=True)[:50]:
            f.write(f"Q: What MCP servers does {name} maintain?\n")
            f.write(f"A: {name} maintains the following MCP servers: {', '.join(c['servers'])}. Their repos have {c['total_stars']} total stars on GitHub.\n\n")

        # General Q&A
        f.write("Q: What is the Model Context Protocol (MCP)?\n")
        f.write("A: The Model Context Protocol (MCP) is an open standard for connecting AI models to external tools, data sources, and services. It provides a standardized way for AI assistants to interact with the outside world through a server-client architecture.\n\n")

        f.write("Q: How many MCP servers exist?\n")
        f.write(f"A: As of {datetime.now().strftime('%B %Y')}, there are approximately {len(servers)} MCP servers tracked on GitHub, with more listed on mcpservers.org and in various package registries.\n\n")
    file_count += 1

    # 10. Newsletter-style writing samples
    log.info("Generating newsletter-style corpus...")
    with open(CORPUS_DIR / "newsletter_samples.txt", "w") as f:
        # Generate sample newsletter content from data
        top_servers = sorted(servers, key=lambda x: x['stars'], reverse=True)[:10]
        f.write("FEATURED MCP SERVERS THIS WEEK\n\n")
        for s in top_servers:
            f.write(f"THIS WEEK'S FEATURED SERVER -- {s['name']}\n\n")
            f.write(f"What it does: {s['description'] or 'An MCP server for extending AI capabilities.'}\n")
            f.write(f"Why it matters: With {s['stars']} stars, {s['name']} is one of the most popular MCP servers in the ecosystem. ")
            f.write(f"Built in {s['language'] or 'multiple languages'}, it demonstrates the growing adoption of MCP.\n")
            f.write(f"GitHub: {s['url']}\n\n")

        # Ecosystem overview
        total_stars = sum(s['stars'] for s in servers)
        org_count = sum(1 for c in companies.values() if c['type'] == 'Organization')
        f.write("MCP ECOSYSTEM OVERVIEW\n\n")
        f.write(f"The MCP ecosystem continues to grow rapidly. There are now {len(servers)} MCP-related repositories on GitHub, ")
        f.write(f"maintained by {len(companies)} different organizations and individuals. ")
        f.write(f"These repos have accumulated {total_stars:,} total stars. ")
        f.write(f"{org_count} organizations are actively building MCP servers.\n\n")

        # Package ecosystem
        f.write("PACKAGE ECOSYSTEM\n\n")
        f.write(f"npm hosts {len(npm_packages)} MCP-related packages, making JavaScript/TypeScript the dominant language for MCP development. ")
        f.write(f"PyPI lists {len(pypi_packages)} Python MCP packages, showing strong adoption in the Python ecosystem as well.\n\n")
    file_count += 1

    log.info(f"Generated {file_count} corpus files in {CORPUS_DIR}/")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(filepath, rows, fieldnames):
    """Write a list of dicts to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote {len(rows)} rows to {filepath}")


def write_sponsorship_report(targets, filepath):
    """Write sponsorship targets markdown report."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("# MCP Sponsorship Targets\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("Ranked by: total stars + 100 * number of MCP servers\n\n")
        f.write("| Rank | Organization | Servers | Stars | Score | Top Repos |\n")
        f.write("|------|-------------|---------|-------|-------|----------|\n")
        for i, t in enumerate(targets[:30], 1):
            f.write(f"| {i} | [{t['company']}]({t['url']}) | {t['num_servers']} | {t['total_stars']} | {t['score']} | {t['servers']} |\n")
        f.write("\n## Notes\n\n")
        f.write("- Organizations with multiple MCP servers indicate deeper investment in the ecosystem\n")
        f.write("- High star count suggests community trust and visibility\n")
        f.write("- Consider reaching out to top 10 as primary sponsorship candidates\n")
    log.info(f"Wrote sponsorship report to {filepath}")


def write_opportunities_report(servers, companies, npm_packages, filepath):
    """Write ecosystem opportunities/gaps report."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Analyze gaps
    languages = {}
    categories = {}
    for s in servers:
        lang = s.get("language", "Unknown") or "Unknown"
        languages[lang] = languages.get(lang, 0) + 1
        for topic in s.get("topics", "").split(","):
            topic = topic.strip()
            if topic:
                categories[topic] = categories.get(topic, 0) + 1

    with open(filepath, "w") as f:
        f.write("# MCP Ecosystem Opportunities\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Language Distribution\n\n")
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:15]:
            f.write(f"- **{lang}**: {count} servers\n")

        f.write("\n## Popular Categories\n\n")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:20]:
            f.write(f"- **{cat}**: {count} servers\n")

        f.write("\n## Potential Gaps\n\n")
        f.write("- Languages with few MCP servers may represent underserved communities\n")
        f.write("- Categories with high demand but few servers are opportunities\n")
        f.write("- Enterprise-focused MCP servers (auth, compliance, monitoring) are still emerging\n")
        f.write("- Mobile and edge MCP deployments are largely unexplored\n")
    log.info(f"Wrote opportunities report to {filepath}")


def write_protocol_changes(spec_pages, filepath):
    """Write protocol changes report from spec data."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("# MCP Protocol Changes & Spec Overview\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for page in spec_pages:
            f.write(f"## {page['title']}\n")
            f.write(f"Source: {page['url']}\n\n")
            # Write first 2000 chars as summary
            content = page['content'][:2000]
            f.write(content)
            if len(page['content']) > 2000:
                f.write("\n\n[... truncated ...]\n")
            f.write("\n\n---\n\n")
    log.info(f"Wrote protocol changes to {filepath}")


# ---------------------------------------------------------------------------
# Qwen analysis (via ollama, if available)
# ---------------------------------------------------------------------------

def analyze_with_qwen(data_summary):
    """Use Qwen via ollama to analyze the collected data and generate insights.
    Falls back gracefully if ollama is not available."""
    try:
        resp = requests.post("http://localhost:11434/api/generate", json={
            "model": "qwen3.5:9b",
            "prompt": f"""Analyze this MCP ecosystem data and provide:
1. Key trends in the MCP ecosystem
2. Most promising sponsorship targets for an MCP newsletter
3. Emerging opportunities and gaps
4. Notable companies and their MCP strategies

Data summary:
{data_summary}

Provide a structured analysis in markdown format.""",
            "stream": False,
        }, timeout=120)
        if resp.ok:
            result = resp.json().get("response", "")
            return result
    except Exception as e:
        log.info(f"Qwen analysis not available (ollama not running): {e}")
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("MCP Ecosystem Researcher - Starting data collection")
    log.info("=" * 60)

    # Collect data from all sources
    official_repos = collect_official_repos()
    topic_repos = collect_mcp_server_repos()
    search_repos = collect_mcp_search_repos()
    npm_packages = collect_npm_packages()
    pypi_packages = collect_pypi_packages()
    mcpservers_list = collect_mcpservers_org()
    spec_pages = collect_mcp_spec()

    reddit_mcp = collect_reddit_posts("MCPservers")
    reddit_claude = collect_reddit_posts("ClaudeAI", query="MCP")
    all_reddit = reddit_mcp + reddit_claude

    blog_articles = collect_anthropic_blog()

    # Process repos into structured data
    servers, companies, contacts = process_all_repos(official_repos, topic_repos, search_repos)
    sponsorship_targets = analyze_sponsorship_targets(companies, servers)

    # Write structured output (CSV/JSON)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    write_csv(OUTPUT_DIR / "servers.csv", servers, [
        "name", "full_name", "description", "url", "stars", "forks",
        "language", "topics", "updated", "owner", "owner_type", "license",
    ])

    company_rows = [
        {"name": name, "type": c["type"], "url": c["url"],
         "num_servers": len(c["servers"]), "servers": ", ".join(c["servers"]),
         "total_stars": c["total_stars"]}
        for name, c in companies.items()
    ]
    write_csv(OUTPUT_DIR / "companies.csv", company_rows, [
        "name", "type", "url", "num_servers", "servers", "total_stars",
    ])

    contact_rows = [
        {"name": name, "type": c["type"], "github": c["github"],
         "repos": ", ".join(c["repos"])}
        for name, c in contacts.items()
    ]
    write_csv(OUTPUT_DIR / "contacts.csv", contact_rows, [
        "name", "type", "github", "repos",
    ])

    write_sponsorship_report(sponsorship_targets, OUTPUT_DIR / "sponsorship_targets.md")
    write_opportunities_report(servers, companies, npm_packages, OUTPUT_DIR / "opportunities.md")
    write_protocol_changes(spec_pages, OUTPUT_DIR / "protocol_changes.md")

    # Generate training corpus
    generate_corpus(servers, companies, contacts, npm_packages, pypi_packages,
                    spec_pages, all_reddit, blog_articles, mcpservers_list)

    # Try Qwen analysis if available
    summary = (
        f"Total servers: {len(servers)}\n"
        f"Total companies: {len(companies)}\n"
        f"npm packages: {len(npm_packages)}\n"
        f"PyPI packages: {len(pypi_packages)}\n"
        f"Reddit posts: {len(all_reddit)}\n"
        f"Spec pages: {len(spec_pages)}\n"
        f"Top 5 by stars: {', '.join(s['name'] for s in sorted(servers, key=lambda x: x['stars'], reverse=True)[:5])}\n"
    )
    qwen_analysis = analyze_with_qwen(summary)
    if qwen_analysis:
        with open(OUTPUT_DIR / "qwen_analysis.md", "w") as f:
            f.write("# MCP Ecosystem Analysis (Qwen)\n\n")
            f.write(qwen_analysis)
        log.info("Qwen analysis saved to results/qwen_analysis.md")

    # Final summary
    log.info("=" * 60)
    log.info("MCP Ecosystem Research Complete!")
    log.info(f"  Servers found:     {len(servers)}")
    log.info(f"  Companies found:   {len(companies)}")
    log.info(f"  Contacts found:    {len(contacts)}")
    log.info(f"  npm packages:      {len(npm_packages)}")
    log.info(f"  PyPI packages:     {len(pypi_packages)}")
    log.info(f"  Reddit posts:      {len(all_reddit)}")
    log.info(f"  Spec pages:        {len(spec_pages)}")
    log.info(f"  Blog articles:     {len(blog_articles)}")
    log.info(f"  mcpservers.org:    {len(mcpservers_list)}")
    log.info(f"  Structured data:   {OUTPUT_DIR}/")
    log.info(f"  Training corpus:   {CORPUS_DIR}/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
