# PokeCounterBot 🎮

A Discord bot that automates a Pokémon GO CP counting game in `#poke-counter` by running OCR on uploaded screenshots.

Designed to run 24/7 on remote Linux servers or lightweight VPS containers with minimal resource overhead.

---

## 🌟 Features

- **Automated CP Detection**: Uses an OpenCV + Tesseract OCR pipeline specifically tuned for Pokémon GO screenshot resolutions and lighting variants (day, sunset, night skies).
- **Stateless Count Recovery**: Reads its own message history in `#poke-counter` upon restart:
  - If the latest bot message is `{N} ✅` $\rightarrow$ expects `N + 1`.
  - If the latest bot message is `{N} ❌ ... begin at 10` or no past bot messages $\rightarrow$ starts at `10`.
- **Zero Database Required**: Zero external database dependency—the Discord channel's message history acts as the single source of truth.
- **Async & Non-Blocking**: Runs OCR in worker threads to prevent blocking the Discord WebSocket heartbeat.
- **Remote Server Ready**: Comes with a production-ready `Dockerfile`, `docker-compose.yml`, and `systemd` service unit.

---

## 🕹️ Game Rules

1. Users upload Pokémon GO stat screenshots in `#poke-counter`.
2. The bot extracts the Pokémon's CP:
   - If CP matches the required number: Bot reacts with `✅` and posts `{CP} ✅`.
   - If CP does not match: Bot reacts with `❌` and posts `{Extracted_CP} ❌ Wrong CP, begin at 10.`
   - If the image cannot be recognized as a Pokémon CP: Bot reacts with `❓` and warns the user without breaking the current streak.
3. Users can type `!count_status` in `#poke-counter` to see the current count and next expected CP.

---

## ⚙️ Configuration (`.env`)

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DISCORD_TOKEN` | *Required* | Your Discord Bot Token from the Developer Portal. |
| `CHANNEL_NAME` | `poke-counter` | Channel name the bot listens to. |
| `CHANNEL_ID` | *Optional* | Exact Discord Channel ID (takes precedence over name). |
| `STARTING_CP` | `10` | The base number to count from when starting or resetting. |
| `ALLOW_CONSECUTIVE_COUNTS` | `true` | Allow the same user to post multiple consecutive counts. |
| `TESSERACT_CMD` | *Optional* | Custom path to the `tesseract` binary if not in `PATH`. |

---

## 🤖 Discord Developer Portal Setup

1. Go to [Discord Developer Portal](https://discord.com/developers/applications) and click **New Application**.
2. Go to the **Bot** tab:
   - Click **Reset Token** to copy your token (paste into `.env` as `DISCORD_TOKEN`).
   - Under **Privileged Gateway Intents**, enable:
     - ✅ **Message Content Intent** *(Required to read messages and image attachments)*
     - ✅ **Server Members Intent** *(Optional)*
3. Go to **OAuth2** $\rightarrow$ **URL Generator**:
   - Scopes: `bot`
   - Bot Permissions:
     - `Read Messages/View Channels`
     - `Send Messages`
     - `Read Message History`
     - `Add Reactions`
     - `Attach Files`
4. Copy the generated URL to invite the bot to your Discord server.

---

## 🚀 Remote Server Deployment

### Method 1: Docker Compose (Recommended)

1. Clone your repo onto the server:
   ```bash
   git clone <YOUR_GITHUB_REPO_URL> /opt/PokeCounterBot
   cd /opt/PokeCounterBot
   ```

2. Create your `.env` file:
   ```bash
   cp .env.example .env
   nano .env
   ```

3. Launch in the background:
   ```bash
   docker compose up -d --build
   ```

4. View live logs:
   ```bash
   docker compose logs -f
   ```

To stop or restart:
```bash
docker compose restart
docker compose down
```

---

### Method 2: Systemd on Ubuntu / Debian VPS

1. Install dependencies on your server:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-venv python3-pip tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 git
   ```

2. Clone and setup virtual environment:
   ```bash
   git clone <YOUR_GITHUB_REPO_URL> /opt/PokeCounterBot
   cd /opt/PokeCounterBot
   python3 -m venv venv
   ./venv/bin/pip install --upgrade pip
   ./venv/bin/pip install -r requirements.txt
   ```

3. Configure `.env`:
   ```bash
   cp .env.example .env
   nano .env
   ```

4. Install the systemd service:
   ```bash
   sudo cp pokecounterbot.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable pokecounterbot
   sudo systemctl start pokecounterbot
   ```

5. Check status and logs:
   ```bash
   sudo systemctl status pokecounterbot
   sudo journalctl -u pokecounterbot -f
   ```

---

## 🧪 Local Testing

To run the automated unit test suite (testing both OCR and counting game logic):

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
