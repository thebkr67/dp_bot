# 🧠 Telegram ChatGPT Bot Template (Aiogram + GPT-3.5 Turbo)

A ready-to-use Telegram chatbot template built with **Aiogram (v3)** and powered by **OpenAI's GPT-3.5 Turbo**. You can easily customize the system prompt to adapt the chatbot for various use-cases such as customer support, tutoring, virtual assistant, and more.

---

## 🚀 Features

- Fast and asynchronous handling using `Aiogram`
- Modular design with environment config
- ChatGPT 3.5 Turbo responses via OpenAI API
- Customizable prompt per use case
- Basic logging and error handling
- Built-in support for future scalability

---

## 🧱 Project Structure

```text
telegram-chatgpt-bot/
├── main.py             # Main bot logic
├── requirements.txt   # List of Python dependencies
└── .env               # Your secrets (OpenAI key, bot token)
```

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/Asit2003/Telegram-Bot-Using-ChatGPT.git
cd Telegram-Bot-Using-ChatGPT
```

### 2. Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env` File

Add your credentials inside a file called `.env`:

```env
BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
```

---

## 🤖 How to Create a Telegram Bot Using BotFather

1. Open Telegram and search for [@BotFather](https://t.me/BotFather).
2. Start a chat and use the command:
   ```
   /newbot
   ```
3. Follow the prompts:
    - Choose a name for your bot (e.g., `My GPT Bot`)
    - Choose a username that ends in `bot` (e.g., `my_gpt_bot`)
4. BotFather will generate a **token** like this:
   ```
   123456789:ABCdefGhIJKlmNoPQRsTUvwxyz
   ```
5. Copy this token and paste it into your `.env` file as:
   ```env
   BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUvwxyz
   ```
6. Your bot is now created! Use `/setdescription`, `/setabouttext`, and `/setuserpic` to customize it further.

---


## 💬 How It Works

- The bot listens to user messages on Telegram.
- It sends the user's input to OpenAI ChatGPT 3.5 Turbo using the `openai.ChatCompletion.create` endpoint.
- The model responds based on the `SYSTEM_PROMPT`.
- The bot sends the reply back to the user.

---

## ▶️ Run the Bot

```bash
python main.py
```

---

## 🧠 Example Prompt Ideas

Change the `SYSTEM_PROMPT` in `.env` to:

| Use Case       | Example Prompt |
|----------------|----------------|
| Tutor          | You are a patient math tutor who explains things simply. |
| Support Agent  | You are a polite customer support agent for a fintech startup. |
| Therapist      | You are an empathetic mental health assistant. |
| Friend         | You are a fun and casual chat buddy. |

---

## 📦 requirements.txt

```
aiogram==3.0.0b7
openai
python-dotenv
```


## 📌 Notes

- This bot uses polling. For production, consider using webhook-based deployment.
- To switch models (e.g., GPT-4), update the model name.
- Always rotate API keys and secure `.env` files in production.

---

## 📞 Support

For issues or customization help, open a GitHub issue.

---

