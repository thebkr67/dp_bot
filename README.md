# Telegram Bot с DeepSeek API

Telegram-бот с интеграцией DeepSeek API для ответов на вопросы пользователей.

## Функции

- **Для пользователей:**
  - Отправка вопросов и получение ответов от DeepSeek
  - Кнопка "Помощь"
  - Сохранение статистики активности

- **Для админа (ID: 1129806592):**
  - Команда `/stats` - количество пользователей
  - Команда `/broadcast` - рассылка сообщений всем пользователям
  - Команда `/activity` - активность по времени суток
  - Кнопки "Статистика" и "Рассылка"

## Деплой на Railway

### 1. Подготовка репозитория
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

### 2. Создание проекта на Railway
1. Зайдите на [railway.app](https://railway.app)
2. Создайте новый проект
3. Выберите "Deploy from GitHub repo"
4. Подключите ваш репозиторий

### 3. Настройка переменных окружения
В Railway Dashboard → Variables добавьте:
- `TELEGRAM_BOT_TOKEN` = ваш токен Telegram бота
- `DEEPSEEK_API_KEY` = ваш ключ DeepSeek API

### 4. Деплой
Railway автоматически:
- Установит зависимости из `requirements.txt`
- Запустит бота согласно `Procfile`
- Покажет логи запуска

## Локальный запуск

```bash
# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt

# Запуск бота
python bot.py
```

## Структура проекта

```
├── bot.py              # Основной код бота
├── requirements.txt    # Зависимости Python
├── Procfile           # Конфигурация для Railway
├── .gitignore         # Исключения для Git
├── README.md          # Документация
└── users.json         # База пользователей (создается автоматически)
```

## Команды бота

- `/start` - запуск бота
- `/stats` - статистика пользователей (только админ)
- `/broadcast <текст>` - рассылка сообщения (только админ)
- `/activity` - активность по часам (только админ) 