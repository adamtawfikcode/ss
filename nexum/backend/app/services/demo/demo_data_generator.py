"""
Nexum Demo Data Generator — Massive Realistic Graph Network.

Generates hundreds of interconnected nodes across all 8 node types
with natural-looking synthetic data covering Islamic studies, tech,
science, and general YouTube content ecosystems.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
# Seed Data — realistic names, titles, topics drawn from real content
# ═══════════════════════════════════════════════════════════════════════

CHANNELS = [
    {"name": "Bayyinah Institute", "platform": "youtube", "category": "islamic_studies", "subs": 1_840_000, "country": "US"},
    {"name": "Yaqeen Institute", "platform": "youtube", "category": "islamic_studies", "subs": 980_000, "country": "US"},
    {"name": "Mufti Menk", "platform": "youtube", "category": "islamic_studies", "subs": 4_200_000, "country": "ZW"},
    {"name": "Al-Madina Institute", "platform": "youtube", "category": "islamic_studies", "subs": 320_000, "country": "US"},
    {"name": "Quran Weekly", "platform": "youtube", "category": "islamic_studies", "subs": 510_000, "country": "US"},
    {"name": "Islamic Guidance", "platform": "youtube", "category": "islamic_studies", "subs": 2_100_000, "country": "GB"},
    {"name": "FreeCodeCamp", "platform": "youtube", "category": "tech", "subs": 9_800_000, "country": "US"},
    {"name": "Fireship", "platform": "youtube", "category": "tech", "subs": 3_200_000, "country": "US"},
    {"name": "NetworkChuck", "platform": "youtube", "category": "tech", "subs": 4_100_000, "country": "US"},
    {"name": "The Linux Experiment", "platform": "youtube", "category": "tech", "subs": 420_000, "country": "FR"},
    {"name": "DistroTube", "platform": "youtube", "category": "tech", "subs": 310_000, "country": "US"},
    {"name": "Kurzgesagt", "platform": "youtube", "category": "science", "subs": 22_400_000, "country": "DE"},
    {"name": "Veritasium", "platform": "youtube", "category": "science", "subs": 15_600_000, "country": "US"},
    {"name": "3Blue1Brown", "platform": "youtube", "category": "science", "subs": 6_200_000, "country": "US"},
    {"name": "SmarterEveryDay", "platform": "youtube", "category": "science", "subs": 11_000_000, "country": "US"},
    {"name": "Al Jazeera English", "platform": "youtube", "category": "news", "subs": 13_500_000, "country": "QA"},
    {"name": "TRT World", "platform": "youtube", "category": "news", "subs": 4_900_000, "country": "TR"},
    {"name": "WION", "platform": "youtube", "category": "news", "subs": 8_700_000, "country": "IN"},
    {"name": "Middle East Eye", "platform": "youtube", "category": "news", "subs": 2_100_000, "country": "GB"},
    {"name": "Arabic Pod 101", "platform": "youtube", "category": "language", "subs": 680_000, "country": "US"},
    {"name": "Spoken Arabic", "platform": "youtube", "category": "language", "subs": 210_000, "country": "EG"},
    {"name": "Linus Tech Tips", "platform": "youtube", "category": "tech", "subs": 16_200_000, "country": "CA"},
    {"name": "Gamers Nexus", "platform": "youtube", "category": "tech", "subs": 3_800_000, "country": "US"},
    {"name": "Two Minute Papers", "platform": "youtube", "category": "science", "subs": 1_600_000, "country": "HU"},
    {"name": "Andrej Karpathy", "platform": "youtube", "category": "tech", "subs": 890_000, "country": "US"},
    {"name": "Traversy Media", "platform": "youtube", "category": "tech", "subs": 2_200_000, "country": "US"},
    {"name": "Arabic Calligraphy Art", "platform": "youtube", "category": "art", "subs": 140_000, "country": "SA"},
    {"name": "Atlas Pro", "platform": "youtube", "category": "science", "subs": 2_400_000, "country": "US"},
    {"name": "History With Hilbert", "platform": "youtube", "category": "history", "subs": 560_000, "country": "US"},
    {"name": "NileRed", "platform": "youtube", "category": "science", "subs": 7_800_000, "country": "CA"},
    {"name": "Lex Fridman", "platform": "youtube", "category": "tech", "subs": 4_500_000, "country": "US"},
    {"name": "TechLinked", "platform": "youtube", "category": "tech", "subs": 2_900_000, "country": "CA"},
]

_VIDEO_TEMPLATES = {
    "islamic_studies": [
        "Linguistic Miracles of Surah {surah}",
        "Tafsir Session: {surah} - Verses {v1}-{v2}",
        "Understanding {concept} in Classical Arabic",
        "The Story of Prophet {prophet} — Full Lecture",
        "Sahih {collection}: Chapter on {topic}",
        "Why {concept} Matters Today — Khutbah",
        "Arabic Grammar Deep Dive: {grammar_topic}",
        "Ramadan Series — Night {n}: {surah}",
        "Q&A: Common Misconceptions About {topic}",
        "Seerah Ep. {n}: The Migration to {place}",
        "How the Quran Changed {historical_figure}'s Life",
        "Spiritual Reflections on {concept}",
        "Tajweed Masterclass: Rules of {tajweed_rule}",
    ],
    "tech": [
        "{lang} in 100 Seconds",
        "I Built a {project} with {framework}",
        "Linux Tip: {linux_topic} Like a Pro",
        "{distro} vs {distro2} — Which Is Better in 2025?",
        "Stop Using {old_tech}! Try {new_tech} Instead",
        "How {company} Uses {technology} at Scale",
        "The {adj} Way to Learn {lang}",
        "Setting Up {tool} on {os} — Complete Guide",
        "Why I Switched to {distro} After 10 Years",
        "Building a Home Lab: {hardware} Edition",
        "{framework} Tutorial for Beginners (2025)",
        "Machine Learning with {ml_lib} — From Zero to Hero",
        "I Automated My Entire {workflow} with Python",
        "RTX {gpu} Benchmarks: Is It Worth the Upgrade?",
    ],
    "science": [
        "What Happens When {phenomenon}?",
        "The Most {adj} Thing in the Universe",
        "How {organism} Actually Works",
        "The Math Behind {concept}",
        "We Solved {problem} (After {n} Years)",
        "Why {misconception} Is Wrong",
        "The Absurd Physics of {topic}",
        "Can We Actually Build {megaproject}?",
        "{n} Things You Didn't Know About {topic}",
        "The Hidden Chemistry of {everyday_thing}",
    ],
    "news": [
        "{region} Update: {event_type} in {country}",
        "Analysis: What {leader}'s Decision Means for {region}",
        "Breaking Down the {topic} Crisis — Full Report",
        "{country}'s New {policy} Explained",
        "Inside the {conflict}: Report from {city}",
        "Expert Panel: The Future of {topic}",
    ],
    "language": [
        "Arabic Lesson {n}: {grammar_topic}",
        "{n} Essential {dialect} Phrases for Beginners",
        "How to Pronounce {letter} Correctly",
        "Conversational Arabic: At the {place}",
        "Reading Practice: {text_type} in Modern Standard Arabic",
    ],
    "art": [
        "Thuluth Calligraphy: Writing {phrase}",
        "The Art of {script} Script — Full Tutorial",
        "How to Write {surah} in Naskh Style",
    ],
    "history": [
        "The Rise and Fall of {empire}",
        "How {civilization} Changed the World",
        "{year}: The Year That Changed Everything",
        "The Untold Story of {event}",
    ],
}

_FILL = {
    "surah": ["Al-Baqarah", "Yasin", "Ar-Rahman", "Al-Kahf", "Al-Mulk", "An-Nisa", "Al-Anfal", "Maryam", "Al-Isra", "Taha", "Al-Furqan", "Al-Ahzab", "Az-Zumar", "Ghafir", "Ad-Dukhan", "Al-Hujurat"],
    "concept": ["Tawakkul", "Ihsan", "Barakah", "Taqwa", "Sabr", "Shukr", "Istighfar", "Tawhid", "Adab", "Zuhd", "Muraqabah"],
    "prophet": ["Yusuf", "Musa", "Ibrahim", "Nuh", "Isa", "Dawud", "Sulayman", "Ayyub", "Yunus"],
    "collection": ["Bukhari", "Muslim", "Tirmidhi", "Abu Dawud", "An-Nasai"],
    "topic": ["Patience", "Charity", "Fasting", "Prayer", "Justice", "Knowledge Seeking", "Family Ties", "Trade Ethics", "Water Rights"],
    "grammar_topic": ["Ism al-Fa'il", "Mubtada and Khabar", "Haal Constructions", "Idafa Chains", "Maf'ul Mutlaq", "Jawab al-Shart"],
    "tajweed_rule": ["Idgham", "Ikhfa", "Iqlab", "Qalqalah", "Ghunna", "Madd Lazim"],
    "place": ["Madinah", "Abyssinia", "Ta'if", "Hudaybiyyah"],
    "historical_figure": ["Umar ibn al-Khattab", "Khalid ibn al-Walid", "Salahuddin"],
    "lang": ["Rust", "Go", "Python", "TypeScript", "Zig", "Elixir", "Haskell", "C++", "Kotlin", "Swift"],
    "framework": ["Next.js", "Svelte", "FastAPI", "HTMX", "Astro", "SolidJS", "Tauri", "React Native"],
    "project": ["Real-Time Chat App", "Git Client", "Password Manager", "Music Player", "Web Scraper", "CLI Tool"],
    "linux_topic": ["Systemd Services", "Btrfs Snapshots", "Wayland Compositors", "PipeWire Audio", "ZFS Pools", "Podman Containers"],
    "distro": ["Arch", "Fedora", "NixOS", "Debian", "openSUSE", "Void Linux", "Gentoo"],
    "distro2": ["Ubuntu", "Pop!_OS", "EndeavourOS", "Manjaro", "Linux Mint", "Garuda"],
    "old_tech": ["Docker Compose v1", "Webpack", "REST APIs", "jQuery", "Moment.js"],
    "new_tech": ["Podman", "Vite", "tRPC", "Solid.js", "Temporal"],
    "company": ["Google", "Anthropic", "Cloudflare", "Vercel", "Supabase"],
    "technology": ["WebAssembly", "Edge Computing", "Vector Databases", "gRPC", "eBPF"],
    "tool": ["Neovim", "Tmux", "Zsh", "Alacritty", "Hyprland", "Nix"],
    "os": ["Arch Linux", "Fedora 41", "NixOS", "macOS Sequoia"],
    "hardware": ["Proxmox Cluster", "Raspberry Pi 5", "Mini-PC Firewall"],
    "ml_lib": ["PyTorch", "JAX", "scikit-learn", "Hugging Face Transformers"],
    "workflow": ["Email Inbox", "Server Monitoring", "Media Library", "Backup Pipeline"],
    "gpu": ["5070 Ti", "5080", "4090", "4070 Super"],
    "adj": ["Beautiful", "Dangerous", "Incredible", "Terrifying", "Mysterious", "Fastest", "Simplest"],
    "phenomenon": ["You Fall Into a Neutron Star", "Two Galaxies Collide", "A Photon Enters Your Eye", "The Earth Stops Spinning"],
    "organism": ["Your Immune System", "Tardigrades", "The Human Brain", "Coral Reefs", "Mushroom Networks"],
    "problem": ["The Three-Body Problem", "Room-Temperature Superconductivity", "Protein Folding"],
    "misconception": ["Evolution Is Random", "Glass Is a Liquid", "Lightning Never Strikes Twice"],
    "megaproject": ["A Space Elevator", "A Dyson Sphere", "Terraforming Mars"],
    "everyday_thing": ["Cooking an Egg", "Soap Bubbles", "Your Morning Coffee"],
    "region": ["Middle East", "Southeast Asia", "East Africa", "Central Asia", "North Africa"],
    "event_type": ["Diplomatic Breakthrough", "Economic Reforms", "Elections", "Protests", "Peace Talks"],
    "country": ["Türkiye", "Egypt", "Morocco", "Indonesia", "Malaysia", "Jordan", "Qatar", "UAE", "Kazakhstan"],
    "leader": ["the Prime Minister", "the President", "the Chancellor"],
    "policy": ["Energy Transition Plan", "Tech Sovereignty Bill", "Education Reform"],
    "conflict": ["Refugee Crisis", "Water Dispute", "Trade Standoff"],
    "city": ["Istanbul", "Beirut", "Amman", "Doha", "Kuala Lumpur"],
    "dialect": ["Egyptian Arabic", "Levantine", "Gulf Arabic", "Moroccan Darija"],
    "letter": ["'Ayn (ع)", "Ḥa (ح)", "Qaf (ق)", "Ḍad (ض)", "Ghayn (غ)"],
    "text_type": ["News Article", "Short Story", "Poetry", "Contract"],
    "phrase": ["Bismillah ar-Rahman ar-Rahim", "La Hawla wa La Quwwata", "SubhanAllah"],
    "script": ["Diwani", "Thuluth", "Ruq'ah", "Nastaliq", "Kufic"],
    "empire": ["the Abbasid Caliphate", "the Ottoman Empire", "the Umayyad Dynasty", "Andalusia"],
    "civilization": ["Islamic Golden Age Scholars", "Ancient Egyptian Engineers", "The Mughal Court"],
    "year": ["1258", "1453", "1492", "1683", "1924"],
    "event": ["The House of Wisdom", "The Siege of Constantinople", "The Reconquista"],
    "n": list(range(1, 30)),
    "v1": list(range(1, 50)),
    "v2": list(range(51, 120)),
}

ENTITY_NAMES = [
    ("Nouman Ali Khan", "person"), ("Tim Berners-Lee", "person"), ("Linus Torvalds", "person"),
    ("Ibn Kathir", "person"), ("Imam Ghazali", "person"), ("Andrej Karpathy", "person"),
    ("Geoffrey Hinton", "person"), ("Yann LeCun", "person"), ("Dario Amodei", "person"),
    ("Ilya Sutskever", "person"), ("Demis Hassabis", "person"),
    ("Al-Khawarizmi", "person"), ("Ibn Sina", "person"), ("Ibn Rushd", "person"),
    ("Python", "technology"), ("Rust", "technology"), ("Arch Linux", "technology"),
    ("PyTorch", "technology"), ("TensorFlow", "technology"), ("Kubernetes", "technology"),
    ("Docker", "technology"), ("Qdrant", "technology"), ("Neo4j", "technology"),
    ("PostgreSQL", "technology"), ("FastAPI", "technology"), ("Next.js", "technology"),
    ("WebAssembly", "technology"), ("CUDA", "technology"), ("Whisper", "technology"),
    ("GPT-4", "technology"), ("Claude", "technology"), ("Llama", "technology"),
    ("CLIP", "technology"), ("Stable Diffusion", "technology"),
    ("Google", "organization"), ("Anthropic", "organization"), ("OpenAI", "organization"),
    ("Meta", "organization"), ("Microsoft", "organization"), ("NVIDIA", "organization"),
    ("Cloudflare", "organization"), ("Vercel", "organization"), ("Canonical", "organization"),
    ("Red Hat", "organization"), ("Bayyinah", "organization"), ("Yaqeen Institute", "organization"),
    ("Al Azhar University", "organization"), ("ISNA", "organization"),
    ("Machine Learning", "concept"), ("Neural Networks", "concept"), ("Transformers", "concept"),
    ("Attention Mechanism", "concept"), ("Gradient Descent", "concept"),
    ("Tafsir", "concept"), ("Fiqh", "concept"), ("Hadith Science", "concept"),
    ("Tajweed", "concept"), ("Usul al-Fiqh", "concept"), ("Maqasid al-Shariah", "concept"),
    ("Quantum Computing", "concept"), ("Zero-Knowledge Proofs", "concept"),
    ("Silicon Valley", "place"), ("Madinah", "place"), ("Makkah", "place"),
    ("Istanbul", "place"), ("Cairo", "place"), ("Córdoba", "place"),
    ("Baghdad", "place"), ("Damascus", "place"), ("Kuala Lumpur", "place"),
    ("London", "place"), ("San Francisco", "place"), ("Toronto", "place"),
    ("MIT", "organization"), ("Stanford", "organization"), ("Cambridge", "organization"),
    ("Linux Kernel", "technology"), ("Wayland", "technology"), ("PipeWire", "technology"),
    ("systemd", "technology"), ("Nix", "technology"), ("Hyprland", "technology"),
    ("Neovim", "technology"), ("Zsh", "technology"), ("Git", "technology"),
]

TOPIC_NAMES = [
    "Quranic Linguistics", "Islamic Jurisprudence", "Prophetic Biography",
    "Arabic Grammar", "Tajweed Recitation", "Hadith Authentication",
    "Machine Learning", "Deep Learning", "Natural Language Processing",
    "Computer Vision", "Audio Processing", "Reinforcement Learning",
    "Linux Administration", "Kernel Development", "Desktop Environments",
    "Web Development", "Systems Programming", "DevOps",
    "Astrophysics", "Quantum Mechanics", "Organic Chemistry",
    "Neuroscience", "Climate Science", "Evolutionary Biology",
    "Middle East Geopolitics", "Tech Industry Analysis", "Open Source Movement",
    "Arabic Calligraphy", "Islamic Architecture", "Classical Arabic Literature",
    "GPU Computing", "Edge AI", "Model Compression",
    "Privacy Engineering", "Cryptography", "API Design",
    "Educational Technology", "Digital Humanities", "Computational Linguistics",
    "Data Engineering", "Vector Databases", "Graph Theory",
]

AUTHOR_FIRST = ["Ahmad", "Fatima", "Muhammad", "Aisha", "Omar", "Zahra", "Ibrahim", "Maryam",
                "Ali", "Khadija", "Hassan", "Layla", "Yusuf", "Noor", "Bilal", "Samira",
                "Tariq", "Hana", "Khalid", "Amira", "Sami", "Dina", "Rami", "Sara",
                "Jake", "Emily", "Chris", "Maria", "David", "Sophie", "Michael", "Anna",
                "Ben", "Lisa", "Alex", "Nina", "Max", "Leah", "Ryan", "Chloe",
                "Dev", "Priya", "Raj", "Ananya", "Kai", "Mei", "Jin", "Yuki",
                "Carlos", "Elena", "Marco", "Giulia", "André", "Lucia", "Pedro", "Sofia"]

AUTHOR_SUFFIXES = ["_dev", "_codes", "_learns", "_reads", "_tech", "_ai", "_ml",
                   "_arb", "", "42", "99", "_official", "_real", "2025", "_pro",
                   "_daily", "_notes", "101", "_eng", "_linux", "_yt"]


def _uid(prefix: str, i: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nexum.demo.{prefix}.{i}"))


def _fill_template(template: str) -> str:
    """Fill a template string with random values from _FILL."""
    result = template
    for key, values in _FILL.items():
        tag = "{" + key + "}"
        while tag in result:
            val = random.choice(values)
            result = result.replace(tag, str(val), 1)
    return result


def _random_date(days_back: int = 730) -> datetime:
    return datetime.now(timezone.utc) - timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )


def _random_iso(days_back: int = 730) -> str:
    return _random_date(days_back).isoformat()


class DemoDataGenerator:
    """Generates a complete, large, interconnected demo dataset."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._channels: List[Dict] = []
        self._videos: List[Dict] = []
        self._comments: List[Dict] = []
        self._authors: List[Dict] = []
        self._entities: List[Dict] = []
        self._topics: List[Dict] = []
        self._playlists: List[Dict] = []
        self._segments: List[Dict] = []
        self._edges: List[Dict] = []
        self._generated = False

    # ── Main Generation ──────────────────────────────────────────────

    def generate(
        self,
        num_videos: int = 240,
        num_comments: int = 1800,
        num_authors: int = 350,
    ) -> Dict[str, Any]:
        """Generate the full demo dataset. Returns graph snapshot + stats."""
        if self._generated:
            return self._snapshot()

        self._gen_channels()
        self._gen_topics()
        self._gen_entities()
        self._gen_authors(num_authors)
        self._gen_videos(num_videos)
        self._gen_playlists()
        self._gen_segments()
        self._gen_comments(num_comments)
        self._gen_cross_links()

        # v4.2: Generate deep analytics signals
        from app.services.demo.signal_generators import SignalGenerator
        self._signal_gen = SignalGenerator(
            videos=self._videos,
            channels=self._channels,
            comments=self._comments,
            authors=self._authors,
            entities=self._entities,
            topics=self._topics,
        )
        self._signal_gen.generate_all()

        self._generated = True
        return self._snapshot()

    # ── Channel Generation ───────────────────────────────────────────

    def _gen_channels(self):
        for i, ch in enumerate(CHANNELS):
            cid = _uid("channel", i)
            self._channels.append({
                "id": cid,
                "platform": ch["platform"],
                "platform_id": f"yt:UC{hashlib.md5(ch['name'].encode()).hexdigest()[:22]}",
                "name": ch["name"],
                "url": f"https://www.youtube.com/@{ch['name'].replace(' ', '')}",
                "custom_url": f"@{ch['name'].replace(' ', '')}",
                "description": f"Official channel for {ch['name']}. Subscribe for the latest content.",
                "country": ch["country"],
                "subscriber_count": ch["subs"],
                "total_videos": random.randint(80, 2000),
                "priority_tier": random.choice([1, 2, 3]),
                "is_active": True,
                "category": ch["category"],
                "created_at": _random_iso(1500),
                "updated_at": _random_iso(30),
                "last_crawled_at": _random_iso(14),
            })

    # ── Topic Generation ─────────────────────────────────────────────

    def _gen_topics(self):
        for i, name in enumerate(TOPIC_NAMES):
            self._topics.append({
                "id": _uid("topic", i),
                "name": name,
                "mention_count": random.randint(5, 500),
                "created_at": _random_iso(800),
                "updated_at": _random_iso(30),
            })

    # ── Entity Generation ────────────────────────────────────────────

    def _gen_entities(self):
        for i, (name, etype) in enumerate(ENTITY_NAMES):
            self._entities.append({
                "id": _uid("entity", i),
                "canonical_name": name,
                "entity_type": etype,
                "mention_count": random.randint(3, 800),
                "aliases": [],
                "first_seen_at": _random_iso(700),
                "updated_at": _random_iso(20),
            })

    # ── Author Generation ────────────────────────────────────────────

    def _gen_authors(self, count: int):
        used = set()
        for i in range(count):
            first = random.choice(AUTHOR_FIRST)
            suffix = random.choice(AUTHOR_SUFFIXES)
            name = f"{first}{suffix}"
            while name in used:
                name = f"{first}{random.choice(AUTHOR_SUFFIXES)}{random.randint(1,99)}"
            used.add(name)

            self._authors.append({
                "id": _uid("author", i),
                "display_name": name,
                "platform": "youtube",
                "platform_author_id": f"UC{hashlib.md5(name.encode()).hexdigest()[:22]}",
                "comment_count": random.randint(1, 120),
                "first_seen_at": _random_iso(500),
                "last_seen_at": _random_iso(30),
                "avg_sentiment": round(random.gauss(0.15, 0.35), 3),
                "updated_at": _random_iso(15),
            })

    # ── Video Generation ─────────────────────────────────────────────

    def _gen_videos(self, count: int):
        for i in range(count):
            ch = random.choice(self._channels)
            cat = ch["category"]
            templates = _VIDEO_TEMPLATES.get(cat, _VIDEO_TEMPLATES["tech"])
            title = _fill_template(random.choice(templates))
            duration = random.choice([
                random.randint(60, 600),      # short
                random.randint(600, 2400),     # medium
                random.randint(2400, 7200),    # long
            ])
            views = int(10 ** random.uniform(2.5, 7.2))
            likes = int(views * random.uniform(0.01, 0.08))
            comments = int(views * random.uniform(0.001, 0.01))
            upload_dt = _random_date(600)

            vid = {
                "id": _uid("video", i),
                "platform": "youtube",
                "platform_id": f"youtube:{hashlib.md5(f'vid{i}'.encode()).hexdigest()[:11]}",
                "channel_id": ch["id"],
                "channel_name": ch["name"],
                "title": title,
                "description": f"{title}\n\nIn this video we explore...\n\n#{'#'.join(random.sample([t['name'].replace(' ','') for t in self._topics], k=min(3, len(self._topics))))}",
                "url": f"https://www.youtube.com/watch?v={hashlib.md5(f'vid{i}'.encode()).hexdigest()[:11]}",
                "thumbnail_url": f"https://i.ytimg.com/vi/{hashlib.md5(f'vid{i}'.encode()).hexdigest()[:11]}/maxresdefault.jpg",
                "duration_seconds": duration,
                "view_count": views,
                "like_count": likes,
                "comment_count": comments,
                "language": random.choice(["en", "en", "en", "ar", "ar"]),
                "tags": random.sample([e["canonical_name"] for e in self._entities if e["entity_type"] in ("concept", "technology")], k=min(5, len(self._entities))),
                "categories": [cat],
                "live_status": None,
                "status": "indexed",
                "uploaded_at": upload_dt.isoformat(),
                "created_at": (upload_dt + timedelta(hours=random.randint(1, 72))).isoformat(),
                "updated_at": (upload_dt + timedelta(days=random.randint(1, 60))).isoformat(),
                "last_crawled_at": _random_iso(14),
                "category": cat,
            }
            self._videos.append(vid)

            # Edge: channel UPLOADED video
            self._edges.append({
                "source": ch["id"], "target": vid["id"],
                "source_type": "Channel", "target_type": "Video",
                "edge_type": "UPLOADED", "weight": 1.0,
            })

            # Edge: video DISCUSSES topics (2-4 topics per video)
            cat_topics = [t for t in self._topics if any(
                kw in t["name"].lower() for kw in self._category_keywords(cat)
            )]
            if not cat_topics:
                cat_topics = random.sample(self._topics, k=min(2, len(self._topics)))
            for topic in random.sample(cat_topics, k=min(random.randint(2, 4), len(cat_topics))):
                self._edges.append({
                    "source": vid["id"], "target": topic["id"],
                    "source_type": "Video", "target_type": "Topic",
                    "edge_type": "DISCUSSES", "weight": round(random.uniform(0.4, 1.0), 2),
                })

            # Edge: entity APPEARS_IN video (2-6 entities)
            cat_entities = [e for e in self._entities if any(
                kw in e["entity_type"] for kw in ["technology", "concept", "person"]
            )]
            for entity in random.sample(cat_entities, k=min(random.randint(2, 6), len(cat_entities))):
                self._edges.append({
                    "source": entity["id"], "target": vid["id"],
                    "source_type": "Entity", "target_type": "Video",
                    "edge_type": "APPEARS_IN", "weight": round(random.uniform(0.3, 1.0), 2),
                })

    @staticmethod
    def _category_keywords(cat: str) -> List[str]:
        return {
            "islamic_studies": ["quran", "islamic", "arabic", "grammar", "tajweed", "hadith", "jurisprud", "biography"],
            "tech": ["machine", "deep", "linux", "web", "system", "devops", "gpu", "api", "privacy", "data", "vector", "model"],
            "science": ["physics", "quantum", "chem", "neuro", "climate", "bio", "astro"],
            "news": ["geopol", "industry", "open source"],
            "language": ["arabic", "linguist", "grammar", "comput"],
            "art": ["calligraphy", "islamic", "arabic"],
            "history": ["arabic", "islamic", "classical"],
        }.get(cat, ["tech", "science"])

    # ── Playlist Generation ──────────────────────────────────────────

    def _gen_playlists(self):
        # Group some videos into playlists
        playlist_defs = [
            "Quran Tafsir Series", "Seerah Complete", "Arabic Grammar Fundamentals",
            "Ramadan Lecture Series", "Hadith Sciences", "Linux From Scratch",
            "Machine Learning Bootcamp", "Rust Programming", "Web Dev 2025",
            "Science Explained", "Physics Fundamentals", "Chemistry Deep Dives",
            "Middle East Analysis", "Tech News Weekly", "Calligraphy Tutorials",
            "Home Lab Setup", "Python Mastery", "DevOps Pipeline",
            "Neural Network Architecture", "GPU Computing Guide",
        ]
        vids_by_cat = {}
        for v in self._videos:
            vids_by_cat.setdefault(v["category"], []).append(v)

        for i, title in enumerate(playlist_defs):
            ch = random.choice(self._channels)
            cat_vids = vids_by_cat.get(ch["category"], self._videos)
            items = random.sample(cat_vids, k=min(random.randint(6, 20), len(cat_vids)))

            pid = _uid("playlist", i)
            self._playlists.append({
                "id": pid,
                "channel_id": ch["id"],
                "platform_playlist_id": f"PL{hashlib.md5(f'pl{i}'.encode()).hexdigest()[:32]}",
                "title": title,
                "description": f"Complete playlist: {title}",
                "thumbnail_url": items[0]["thumbnail_url"] if items else None,
                "video_count": len(items),
                "url": f"https://www.youtube.com/playlist?list=PL{hashlib.md5(f'pl{i}'.encode()).hexdigest()[:32]}",
                "created_at": _random_iso(400),
                "updated_at": _random_iso(20),
                "item_ids": [v["id"] for v in items],
            })

            # Edges: playlist CONTAINS video
            for v in items:
                self._edges.append({
                    "source": pid, "target": v["id"],
                    "source_type": "Playlist", "target_type": "Video",
                    "edge_type": "CONTAINS", "weight": 1.0,
                })

            # Edge: channel HAS_PLAYLIST
            self._edges.append({
                "source": ch["id"], "target": pid,
                "source_type": "Channel", "target_type": "Playlist",
                "edge_type": "HAS_PLAYLIST", "weight": 1.0,
            })

    # ── Segment Generation ───────────────────────────────────────────

    def _gen_segments(self):
        for v in self._videos:
            dur = v["duration_seconds"] or 300
            seg_len = random.choice([5, 10, 15])
            n_segs = min(dur // max(seg_len, 1), 60)
            for j in range(n_segs):
                start = j * seg_len
                self._segments.append({
                    "id": _uid(f"segment_{v['id']}", j),
                    "video_id": v["id"],
                    "start_time": float(start),
                    "end_time": float(min(start + seg_len, dur)),
                    "text": f"[Segment {j+1}] Transcribed speech content for {v['title'][:40]}...",
                    "confidence": round(random.uniform(0.75, 0.99), 3),
                    "speaker_label": random.choice([None, "speaker_0", "speaker_1"]),
                    "updated_at": _random_iso(10),
                })

    # ── Comment Generation ───────────────────────────────────────────

    def _gen_comments(self, count: int):
        _comment_templates = [
            "MashaAllah, this is exactly what I needed to hear today.",
            "Subhanallah, the explanation of {topic} was crystal clear.",
            "Can you do a follow-up on {topic}? This was incredible.",
            "I've been studying {topic} for years and this is the best explanation I've found.",
            "The way you broke down {concept} at {t} was perfect.",
            "This changed my perspective entirely. JazakAllahu khairan.",
            "Great video! The {topic} section starting at {t} was mind-blowing.",
            "I tried implementing this in {lang} and it worked perfectly.",
            "Finally someone explains {concept} without dumbing it down.",
            "Been waiting for this video! {topic} is so underrated.",
            "The part about {concept} at {t} needs its own video honestly.",
            "As a {role}, this is incredibly relevant to my work.",
            "I watched this 3 times. Still finding new insights.",
            "Hot take: {concept} isn't as important as everyone thinks.",
            "Anyone else here from the {channel} recommendation?",
            "This is better than most university lectures on {topic}.",
            "Your point about {concept} contradicts what {entity} said though.",
            "I respectfully disagree with the section on {topic}.",
            "Alhamdulillah, sharing this with my study circle.",
            "The production quality keeps getting better!",
        ]
        _reply_templates = [
            "Totally agree with this. {concept} is key.",
            "I had the same thought! Also check out {entity}'s work on this.",
            "Not sure I agree, but interesting perspective.",
            "This is the right take. Most people get {concept} wrong.",
            "JazakAllahu khairan for pointing this out.",
            "Could you elaborate on why you think that?",
            "I tried this approach and can confirm it works.",
            "Respectfully disagree — {concept} is more nuanced than that.",
            "@{author} great point about {topic}.",
            "This thread is more informative than the video lol",
        ]

        topic_names = [t["name"] for t in self._topics]
        entity_names = [e["canonical_name"] for e in self._entities]

        root_comments = []
        for i in range(count):
            vid = random.choice(self._videos)
            author = random.choice(self._authors)

            is_reply = i > count * 0.3 and root_comments and random.random() < 0.4
            parent = random.choice(root_comments) if is_reply else None

            template = random.choice(_reply_templates if is_reply else _comment_templates)
            text = template.format(
                topic=random.choice(topic_names),
                concept=random.choice(entity_names),
                entity=random.choice([e["canonical_name"] for e in self._entities if e["entity_type"] == "person"]),
                t=f"{random.randint(0,59)}:{random.randint(0,59):02d}",
                lang=random.choice(["Python", "Rust", "TypeScript", "Go"]),
                role=random.choice(["software engineer", "student", "researcher", "teacher"]),
                channel=random.choice([c["name"] for c in self._channels]),
                author=random.choice([a["display_name"] for a in self._authors]),
            )

            cid = _uid("comment", i)
            comment = {
                "id": cid,
                "video_id": vid["id"],
                "author_id": author["id"],
                "author_display_name": author["display_name"],
                "parent_comment_id": parent["id"] if parent else None,
                "root_thread_id": (parent.get("root_thread_id") or parent["id"]) if parent else cid,
                "depth_level": (parent["depth_level"] + 1) if parent else 0,
                "text": text,
                "like_count": int(10 ** random.uniform(0, 3.5)),
                "reply_count": 0,
                "language": random.choice(["en", "en", "en", "ar"]),
                "sentiment_score": round(random.gauss(0.2, 0.4), 3),
                "sentiment_label": random.choice(["positive", "positive", "neutral", "negative"]),
                "status": "processed",
                "timestamp_posted": _random_iso(400),
                "entities_mentioned": random.sample(entity_names, k=min(random.randint(0, 3), len(entity_names))),
                "topic_labels": random.sample(topic_names, k=min(random.randint(0, 2), len(topic_names))),
                "updated_at": _random_iso(15),
            }
            self._comments.append(comment)
            if not is_reply:
                root_comments.append(comment)

            # Edge: author WROTE comment
            self._edges.append({
                "source": author["id"], "target": cid,
                "source_type": "CommentAuthor", "target_type": "Comment",
                "edge_type": "WROTE", "weight": 1.0,
            })
            # Edge: comment ON video
            self._edges.append({
                "source": cid, "target": vid["id"],
                "source_type": "Comment", "target_type": "Video",
                "edge_type": "ON", "weight": 1.0,
            })
            # Edge: author COMMENTED_ON video
            self._edges.append({
                "source": author["id"], "target": vid["id"],
                "source_type": "CommentAuthor", "target_type": "Video",
                "edge_type": "COMMENTED_ON", "weight": 1.0,
            })
            # Edge: REPLIES_TO
            if parent:
                self._edges.append({
                    "source": cid, "target": parent["id"],
                    "source_type": "Comment", "target_type": "Comment",
                    "edge_type": "REPLIES_TO", "weight": 1.0,
                })
            # Edge: comment MENTIONS entity
            for ename in comment["entities_mentioned"]:
                ent = next((e for e in self._entities if e["canonical_name"] == ename), None)
                if ent:
                    self._edges.append({
                        "source": cid, "target": ent["id"],
                        "source_type": "Comment", "target_type": "Entity",
                        "edge_type": "MENTIONS", "weight": round(random.uniform(0.5, 1.0), 2),
                    })

    # ── Cross-Links ──────────────────────────────────────────────────

    def _gen_cross_links(self):
        # Entity CO_OCCURS_WITH edges
        for i in range(min(200, len(self._entities) * 3)):
            a, b = random.sample(self._entities, 2)
            self._edges.append({
                "source": a["id"], "target": b["id"],
                "source_type": "Entity", "target_type": "Entity",
                "edge_type": "CO_OCCURS_WITH",
                "weight": round(random.uniform(0.2, 0.9), 2),
            })

        # Channel SHARES_AUDIENCE_WITH
        for i in range(min(40, len(self._channels) * 2)):
            a, b = random.sample(self._channels, 2)
            if a["category"] == b["category"] or random.random() < 0.3:
                self._edges.append({
                    "source": a["id"], "target": b["id"],
                    "source_type": "Channel", "target_type": "Channel",
                    "edge_type": "SHARES_AUDIENCE_WITH",
                    "weight": round(random.uniform(0.3, 0.95), 2),
                })

    # ── Snapshot Output ──────────────────────────────────────────────

    def _snapshot(self) -> Dict[str, Any]:
        nodes = []
        for ch in self._channels:
            nodes.append({"id": ch["id"], "label": ch["name"], "node_type": "Channel", "data": ch})
        for v in self._videos:
            nodes.append({"id": v["id"], "label": v["title"][:60], "node_type": "Video", "data": v})
        for c in self._comments:
            nodes.append({"id": c["id"], "label": c["text"][:50], "node_type": "Comment", "data": c})
        for a in self._authors:
            nodes.append({"id": a["id"], "label": a["display_name"], "node_type": "CommentAuthor", "data": a})
        for e in self._entities:
            nodes.append({"id": e["id"], "label": e["canonical_name"], "node_type": "Entity", "data": e})
        for t in self._topics:
            nodes.append({"id": t["id"], "label": t["name"], "node_type": "Topic", "data": t})
        for p in self._playlists:
            nodes.append({"id": p["id"], "label": p["title"], "node_type": "Playlist", "data": p})
        for s in self._segments[:500]:  # cap for snapshot
            nodes.append({"id": s["id"], "label": f"Seg {s['start_time']:.0f}s", "node_type": "Segment", "data": s})

        # Deduplicate edges
        seen = set()
        edges = []
        for e in self._edges:
            key = (e["source"], e["target"], e["edge_type"])
            if key not in seen:
                seen.add(key)
                edges.append({
                    "source": e["source"],
                    "target": e["target"],
                    "edge_type": e["edge_type"],
                    "weight": e.get("weight", 1.0),
                    "data": {"source_type": e.get("source_type"), "target_type": e.get("target_type")},
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "sampled": False,
            "stats": {
                "channels": len(self._channels),
                "videos": len(self._videos),
                "comments": len(self._comments),
                "authors": len(self._authors),
                "entities": len(self._entities),
                "topics": len(self._topics),
                "playlists": len(self._playlists),
                "segments": len(self._segments),
                "edges": len(edges),
            },
            "signal_stats": self._signal_gen.stats() if hasattr(self, "_signal_gen") else {},
        }

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def channels(self) -> List[Dict]:
        return self._channels

    @property
    def videos(self) -> List[Dict]:
        return self._videos

    @property
    def comments(self) -> List[Dict]:
        return self._comments

    @property
    def authors(self) -> List[Dict]:
        return self._authors

    @property
    def entities(self) -> List[Dict]:
        return self._entities

    @property
    def topics(self) -> List[Dict]:
        return self._topics

    @property
    def playlists(self) -> List[Dict]:
        return self._playlists

    @property
    def segments(self) -> List[Dict]:
        return self._segments

    @property
    def edges(self) -> List[Dict]:
        return self._edges

    @property
    def signals(self):
        """Access the SignalGenerator instance for deep analytics data."""
        return self._signal_gen if hasattr(self, "_signal_gen") else None

    # ── Node Creation ────────────────────────────────────────────────

    def create_node(self, node_type: str, data: Optional[Dict] = None) -> Dict:
        """Create a new demo node of any type with auto-generated fake data."""
        data = data or {}
        nid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        if node_type == "Video":
            ch = random.choice(self._channels)
            cat = ch["category"]
            templates = _VIDEO_TEMPLATES.get(cat, _VIDEO_TEMPLATES["tech"])
            node = {
                "id": nid,
                "platform": "youtube",
                "platform_id": f"youtube:{hashlib.md5(nid.encode()).hexdigest()[:11]}",
                "channel_id": ch["id"],
                "channel_name": ch["name"],
                "title": data.get("title") or _fill_template(random.choice(templates)),
                "url": data.get("url") or f"https://youtube.com/watch?v={hashlib.md5(nid.encode()).hexdigest()[:11]}",
                "duration_seconds": data.get("duration_seconds") or random.randint(120, 3600),
                "view_count": int(10 ** random.uniform(3, 6)),
                "like_count": random.randint(100, 50000),
                "status": "indexed",
                "category": cat,
                "created_at": now,
                "updated_at": now,
            }
            self._videos.append(node)
            self._edges.append({"source": ch["id"], "target": nid, "source_type": "Channel", "target_type": "Video", "edge_type": "UPLOADED", "weight": 1.0})
            # Link to random topics
            for t in random.sample(self._topics, k=min(3, len(self._topics))):
                self._edges.append({"source": nid, "target": t["id"], "source_type": "Video", "target_type": "Topic", "edge_type": "DISCUSSES", "weight": round(random.uniform(0.4, 1.0), 2)})
            return {"id": nid, "label": node["title"][:60], "node_type": "Video", "data": node}

        elif node_type == "Channel":
            node = {
                "id": nid,
                "platform": "youtube",
                "name": data.get("name") or f"{random.choice(AUTHOR_FIRST)}'s Channel",
                "url": data.get("url") or f"https://youtube.com/@demo{hashlib.md5(nid.encode()).hexdigest()[:8]}",
                "subscriber_count": int(10 ** random.uniform(3, 6)),
                "category": random.choice(list(_VIDEO_TEMPLATES.keys())),
                "created_at": now,
                "updated_at": now,
            }
            self._channels.append(node)
            return {"id": nid, "label": node["name"], "node_type": "Channel", "data": node}

        elif node_type == "Comment":
            vid = random.choice(self._videos) if self._videos else None
            author = random.choice(self._authors) if self._authors else None
            node = {
                "id": nid,
                "video_id": vid["id"] if vid else None,
                "author_id": author["id"] if author else None,
                "author_display_name": author["display_name"] if author else "Anonymous",
                "text": data.get("text") or f"Great content on {random.choice([t['name'] for t in self._topics])}!",
                "like_count": random.randint(0, 500),
                "depth_level": 0,
                "sentiment_score": round(random.gauss(0.2, 0.3), 3),
                "status": "processed",
                "timestamp_posted": now,
            }
            self._comments.append(node)
            if vid:
                self._edges.append({"source": nid, "target": vid["id"], "source_type": "Comment", "target_type": "Video", "edge_type": "ON", "weight": 1.0})
            if author:
                self._edges.append({"source": author["id"], "target": nid, "source_type": "CommentAuthor", "target_type": "Comment", "edge_type": "WROTE", "weight": 1.0})
            return {"id": nid, "label": node["text"][:50], "node_type": "Comment", "data": node}

        elif node_type == "CommentAuthor":
            name = data.get("display_name") or f"{random.choice(AUTHOR_FIRST)}{random.choice(AUTHOR_SUFFIXES)}"
            node = {"id": nid, "display_name": name, "platform": "youtube", "comment_count": random.randint(1, 50), "first_seen_at": now, "updated_at": now}
            self._authors.append(node)
            return {"id": nid, "label": name, "node_type": "CommentAuthor", "data": node}

        elif node_type == "Entity":
            name = data.get("canonical_name") or f"{random.choice(['Algorithm', 'Protocol', 'Framework', 'Library', 'Concept'])} {random.choice(['X', 'Y', 'Z', 'Alpha', 'Beta'])}"
            etype = data.get("entity_type") or random.choice(["concept", "technology", "person", "organization", "place"])
            node = {"id": nid, "canonical_name": name, "entity_type": etype, "mention_count": random.randint(1, 100), "first_seen_at": now, "updated_at": now}
            self._entities.append(node)
            return {"id": nid, "label": name, "node_type": "Entity", "data": node}

        elif node_type == "Topic":
            name = data.get("name") or f"{random.choice(['Advanced', 'Introduction to', 'Applied', 'Theoretical'])} {random.choice(['Systems', 'Analysis', 'Methods', 'Engineering'])}"
            node = {"id": nid, "name": name, "mention_count": random.randint(1, 50), "created_at": now, "updated_at": now}
            self._topics.append(node)
            return {"id": nid, "label": name, "node_type": "Topic", "data": node}

        elif node_type == "Playlist":
            ch = random.choice(self._channels)
            title = data.get("title") or f"{random.choice(['Complete', 'Beginner', 'Advanced', 'Full'])} {random.choice([t['name'] for t in self._topics])} Series"
            node = {"id": nid, "channel_id": ch["id"], "title": title, "video_count": random.randint(5, 30), "url": f"https://youtube.com/playlist?list=PL{nid[:16]}", "created_at": now, "updated_at": now}
            self._playlists.append(node)
            self._edges.append({"source": ch["id"], "target": nid, "source_type": "Channel", "target_type": "Playlist", "edge_type": "HAS_PLAYLIST", "weight": 1.0})
            vids = random.sample(self._videos, k=min(random.randint(3, 10), len(self._videos)))
            for v in vids:
                self._edges.append({"source": nid, "target": v["id"], "source_type": "Playlist", "target_type": "Video", "edge_type": "CONTAINS", "weight": 1.0})
            return {"id": nid, "label": title, "node_type": "Playlist", "data": node}

        elif node_type == "Segment":
            vid = random.choice(self._videos) if self._videos else None
            start = random.uniform(0, (vid["duration_seconds"] or 300) - 15) if vid else 0
            node = {"id": nid, "video_id": vid["id"] if vid else None, "start_time": round(start, 1), "end_time": round(start + 10, 1), "text": f"[Demo segment] transcribed content", "confidence": round(random.uniform(0.8, 0.99), 3), "updated_at": now}
            self._segments.append(node)
            return {"id": nid, "label": f"Seg {start:.0f}s", "node_type": "Segment", "data": node}

        raise ValueError(f"Unknown node type: {node_type}")
