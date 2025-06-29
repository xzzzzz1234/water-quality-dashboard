# Flask and related web libraries
Flask
Flask-Cors
gunicorn  # Vercel 推荐的生产环境 WSGI 服务器
requests  # 用于调用 DeepSeek API

# Data handling and processing
pandas
numpy
openpyxl  # pandas 读写 .xlsx 文件需要这个

# Security and Authentication
PyJWT     # 用于 import jwt
passlib   # 用于密码哈希

# AI/ML Libraries (即使只是读取结果，也建议包含)
# 您的数据库文件名包含 'xgboost'，表明项目依赖这些库
scikit-learn
xgboost
