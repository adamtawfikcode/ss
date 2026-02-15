#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Nexum v5.0 — Arch Linux Setup (RTX 5070 Ti GPU)
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; }
info() { echo -e "${CYAN}[→]${NC} $1"; }

echo ""
echo "  ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███╗   ███╗"
echo "  ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║████╗ ████║"
echo "  ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║██╔████╔██║"
echo "  ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║██║╚██╔╝██║"
echo "  ██║ ╚████║███████╗██╔╝ ╚██╗╚██████╔╝██║ ╚═╝ ██║"
echo "  ╚═╝  ╚═══╝╚══════╝╚═╝   ╚═╝ ╚═════╝ ╚═╝     ╚═╝"
echo "  v5.0 — Arch Linux Setup"
echo ""

# ── 1. Check prerequisites ───────────────────────────────────────────

info "Checking prerequisites..."

# Docker
if command -v docker &>/dev/null; then
    log "Docker $(docker --version | grep -oP '\d+\.\d+\.\d+')"
else
    err "Docker not found"
    info "Installing docker..."
    sudo pacman -S --noconfirm docker docker-compose docker-buildx
    sudo systemctl enable --now docker.service
    sudo usermod -aG docker "$USER"
    warn "You were added to the docker group. Log out and back in, then re-run this script."
    exit 1
fi

# Docker Compose (v2 plugin)
if docker compose version &>/dev/null; then
    log "Docker Compose $(docker compose version --short)"
else
    err "Docker Compose plugin not found"
    info "Installing..."
    sudo pacman -S --noconfirm docker-compose
fi

# Docker daemon running
if docker info &>/dev/null; then
    log "Docker daemon running"
else
    err "Docker daemon not running"
    info "Starting..."
    sudo systemctl start docker.service
fi

# ── 2. NVIDIA GPU setup ─────────────────────────────────────────────

info "Checking NVIDIA GPU..."

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    log "GPU: ${GPU_NAME} (${GPU_MEM}, driver ${DRIVER})"
else
    err "nvidia-smi not found"
    info "Installing NVIDIA drivers..."
    warn "For RTX 5070 Ti you need the latest nvidia driver."
    echo ""
    echo "  Option A (recommended): sudo pacman -S nvidia nvidia-utils"
    echo "  Option B (latest):      yay -S nvidia-open-dkms nvidia-utils"
    echo ""
    echo "  Then reboot and re-run this script."
    exit 1
fi

# NVIDIA Container Toolkit
if command -v nvidia-container-toolkit &>/dev/null || pacman -Qi nvidia-container-toolkit &>/dev/null 2>&1; then
    log "NVIDIA Container Toolkit installed"
else
    warn "NVIDIA Container Toolkit not found"
    info "Installing from AUR..."
    echo ""
    echo "  If you have an AUR helper (yay/paru):"
    echo "    yay -S nvidia-container-toolkit"
    echo ""
    echo "  Then configure Docker runtime:"
    echo "    sudo nvidia-ctk runtime configure --runtime=docker"
    echo "    sudo systemctl restart docker"
    echo ""

    # Try auto-install if yay exists
    if command -v yay &>/dev/null; then
        read -rp "  Install with yay now? [Y/n] " yn
        yn=${yn:-Y}
        if [[ "$yn" =~ ^[Yy]$ ]]; then
            yay -S --noconfirm nvidia-container-toolkit
            sudo nvidia-ctk runtime configure --runtime=docker
            sudo systemctl restart docker
            log "NVIDIA Container Toolkit installed and configured"
        fi
    elif command -v paru &>/dev/null; then
        read -rp "  Install with paru now? [Y/n] " yn
        yn=${yn:-Y}
        if [[ "$yn" =~ ^[Yy]$ ]]; then
            paru -S --noconfirm nvidia-container-toolkit
            sudo nvidia-ctk runtime configure --runtime=docker
            sudo systemctl restart docker
            log "NVIDIA Container Toolkit installed and configured"
        fi
    else
        err "No AUR helper found (yay or paru). Install nvidia-container-toolkit manually."
        exit 1
    fi
fi

# Test GPU in Docker
info "Testing GPU access in Docker..."
if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    log "Docker can access GPU"
else
    err "Docker cannot access GPU"
    warn "Try: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    exit 1
fi

# ── 3. Build ─────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
info "Building Nexum (GPU mode, CUDA 12.4)..."
info "This will take 5-10 minutes on first build."
echo ""

docker compose -f docker-compose.yml -f docker-compose.gpu.yml build \
    --build-arg TORCH_INDEX=cu124

log "Build complete"

# ── 4. Start ─────────────────────────────────────────────────────────

echo ""
info "Starting Nexum..."

docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Wait for health checks
info "Waiting for services to be healthy..."
sleep 5

MAX_WAIT=120
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    HEALTHY=$(docker compose ps --format json 2>/dev/null | grep -c '"healthy"' || true)
    TOTAL=$(docker compose ps --format json 2>/dev/null | wc -l || true)
    RUNNING=$(docker compose ps --status running --format json 2>/dev/null | wc -l || true)

    if [ "$RUNNING" -ge 10 ]; then
        break
    fi

    echo -ne "\r  Waiting... ${ELAPSED}s (${RUNNING} containers running)"
    sleep 3
    ELAPSED=$((ELAPSED + 3))
done
echo ""

# Show status
docker compose ps

# ── 5. Done ──────────────────────────────────────────────────────────

echo ""
echo "  ═══════════════════════════════════════════════════════════"
echo ""
log "Nexum is running!"
echo ""
echo "  Frontend:       ${CYAN}http://localhost:3000${NC}"
echo "  API Docs:       ${CYAN}http://localhost:8000/docs${NC}"
echo "  Neo4j Browser:  ${CYAN}http://localhost:7474${NC}  (neo4j / nexum_graph_secret)"
echo "  MinIO Console:  ${CYAN}http://localhost:9001${NC}  (nexum_minio / nexum_minio_secret)"
echo "  Qdrant Dashboard: ${CYAN}http://localhost:6333/dashboard${NC}"
echo ""
echo "  ${YELLOW}To ingest your first video:${NC}"
echo "  Go to http://localhost:3000 → Queue page → paste a URL"
echo ""
echo "  Or via CLI:"
echo "  curl -X POST http://localhost:8000/api/v1/queue/enqueue \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"url\": \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\", \"priority\": \"normal\"}'"
echo ""
echo "  ${YELLOW}Useful commands:${NC}"
echo "  docker compose logs -f api                  # API logs"
echo "  docker compose logs -f worker-processing    # GPU worker logs"
echo "  docker compose -f docker-compose.yml -f docker-compose.gpu.yml down  # Stop"
echo ""
