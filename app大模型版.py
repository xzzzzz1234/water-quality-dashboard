import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from datetime import datetime, timedelta
import jwt
from passlib.hash import pbkdf2_sha256
import sqlite3
import requests
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# --- Configuration ---
DATABASE = r'bo_xgboost_results.db'
app.config['SECRET_KEY'] = 'your_super_secret_key_change_this_in_production'
app.config['JWT_EXPIRATION_DELTA'] = timedelta(hours=1)

# --- DeepSeek AI Configuration ---
DEEPSEEK_API_KEY = 'sk-63c0cb337afc4de0830cc24a85abc929'
DEEPSEEK_API_URL = 'https://api.deepseek.com/chat/completions'


# --- Database Connection Management (保持不变) ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


# --- 数据库初始化函数 (保持不变) ---
def init_db():
    # ... (您提供的init_db函数代码，非常完善，无需改动) ...
    with app.app_context():
        db = get_db()
        cursor = db.cursor()

        # 辅助函数：检查表是否存在某列
        def column_exists(table_name, column_name):
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            return any(col['name'] == column_name for col in columns)

        # 定义每个数据表的理想Schema和新列名，用于迁移
        table_schemas_for_migration = {
            'result_salinity': {
                'new_cols': ['real_value', 'inferred_value'],  # 新表的列名
                'new_schema_sql': '''
                    CREATE TABLE result_salinity (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        real_value REAL NOT NULL,
                        inferred_value REAL NOT NULL
                    )
                '''
            },
            'result_tss': {
                'new_cols': ['real_value', 'inferred_value', 'location'],  # 新表的列名
                'new_schema_sql': '''
                    CREATE TABLE result_tss (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        real_value REAL NOT NULL,
                        inferred_value REAL NOT NULL,
                        location TEXT NOT NULL
                    )
                '''
            },
            'result_turbidity': {
                'new_cols': ['real_value', 'inferred_value', 'location'],  # 新表的列名
                'new_schema_sql': '''
                    CREATE TABLE result_turbidity (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        real_value REAL NOT NULL,
                        inferred_value REAL NOT NULL,
                        location TEXT NOT NULL
                    )
                '''
            }
        }

        # 首先创建用户表（如果不存在），并处理初始用户
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )
        ''')
        cursor.execute("SELECT COUNT(*) FROM user")
        if cursor.fetchone()[0] == 0:
            admin_phone = "18814893256"
            admin_password_hash = pbkdf2_sha256.hash("password")
            user_phone = "13800000002"
            user_password_hash = pbkdf2_sha256.hash("password")

            cursor.execute("INSERT INTO user (phone, password_hash, role) VALUES (?, ?, ?)",
                           (admin_phone, admin_password_hash, "admin"))
            cursor.execute("INSERT INTO user (phone, password_hash, role) VALUES (?, ?, ?)",
                           (user_phone, user_password_hash, "user"))
            db.commit()
            print("初始管理员和用户账户已创建。")
        else:
            print("用户表已存在并已填充。跳过初始用户创建。")

        # 遍历数据表，检查并执行迁移（如果需要）
        for table_name, schema_info in table_schemas_for_migration.items():
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            table_exists = cursor.fetchone()

            if not table_exists:
                # 如果表不存在，则直接创建它（带id列）
                print(f"表 '{table_name}' 不存在。正在创建...")
                cursor.execute(schema_info['new_schema_sql'])
                db.commit()
            else:
                # 如果表存在，检查是否缺少'id'列
                if not column_exists(table_name, 'id'):
                    print(f"表 '{table_name}' 存在但缺少 'id' 列。正在进行数据迁移以添加 'id' 列...")
                    try:
                        db.execute("PRAGMA foreign_keys=OFF;")  # 暂时禁用外键约束
                        db.execute("BEGIN TRANSACTION;")

                        old_table_name = f"{table_name}_old_temp"
                        cursor.execute(f"ALTER TABLE {table_name} RENAME TO {old_table_name}")

                        # 创建新表
                        cursor.execute(schema_info['new_schema_sql'])

                        # 从旧表中获取所有数据。由于旧表可能没有明确的列名，我们通过位置来获取数据。
                        cursor.execute(f"SELECT * FROM {old_table_name}")
                        # 使用list()来确保即使row_factory是sqlite3.Row也能按索引访问
                        old_data = [list(row) for row in cursor.fetchall()]

                        # 获取旧表的列信息以确认列数
                        cursor.execute(f"PRAGMA table_info({old_table_name})")
                        old_cols_info = cursor.fetchall()
                        num_old_cols = len(old_cols_info)

                        # 准备插入语句，使用问号占位符
                        insert_cols_str = ", ".join(schema_info['new_cols'])
                        placeholders = ", ".join(['?'] * len(schema_info['new_cols']))
                        insert_sql = f"INSERT INTO {table_name} ({insert_cols_str}) VALUES ({placeholders})"

                        # 逐行复制数据到新表，通过位置匹配列
                        for row_data in old_data:
                            if len(row_data) != len(schema_info['new_cols']):
                                raise ValueError(
                                    f"旧表 '{old_table_name}' 的行数据列数 ({len(row_data)}) 与新表期望的列数 ({len(schema_info['new_cols'])}) 不匹配。")

                            # 提取值并按顺序插入
                            values_to_insert = [row_data[i] for i in range(len(schema_info['new_cols']))]
                            cursor.execute(insert_sql, values_to_insert)

                        cursor.execute(f"DROP TABLE {old_table_name}")
                        db.execute("COMMIT;")
                        print(f"表 '{table_name}' 的数据迁移已成功完成。")
                    except sqlite3.Error as e:
                        db.execute("ROLLBACK;")
                        print(f"表 '{table_name}' 数据迁移过程中发生数据库错误：{e}。正在回滚更改。")
                    except ValueError as e:
                        db.execute("ROLLBACK;")
                        print(f"表 '{table_name}' 数据迁移过程中发生数据结构不匹配错误：{e}。正在回滚更改。")
                    except Exception as e:
                        db.execute("ROLLBACK;")
                        print(f"表 '{table_name}' 数据迁移过程中发生意外错误：{e}。正在回滚更改。")
                    finally:
                        db.execute("PRAGMA foreign_keys=ON;")  # 重新启用外键约束
                else:
                    print(f"表 '{table_name}' 存在且包含 'id' 列。无需迁移。")

        print("地点数据（悬沙、盐度、浊度）不会自动生成。")
        print(
            "请确保 'bo_xgboost_results.db' 已在 'result_salinity'、'result_tss' 和 'result_turbidity' 表中预先填充了数据。")


# --- JWT & 权限验证辅助函数 (保持不变) ---
def generate_token(user_id, phone, role):
    payload = {
        'user_id': user_id,
        'phone': phone,
        'role': role,
        'exp': datetime.utcnow() + app.config['JWT_EXPIRATION_DELTA']
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')


def decode_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return {'message': 'Token expired', 'error': True}
    except jwt.InvalidTokenError:
        return {'message': 'Invalid token', 'error': True}


def admin_required(f):
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'message': 'Authorization token is missing!'}), 401

        token_parts = auth_header.split(" ")
        token = token_parts[1] if len(token_parts) > 1 and token_parts[0].lower() == 'bearer' else None
        if not token:
            return jsonify({'message': 'Token format is invalid! Expected "Bearer <token>"'}), 401

        decoded_token = decode_token(token)
        if decoded_token.get('error'):
            return jsonify(decoded_token), 401

        if decoded_token.get('role') != 'admin':
            return jsonify({'message': 'Admin access required!'}), 403

        request.user_id = decoded_token.get('user_id')
        request.user_role = decoded_token.get('role')
        return f(*args, **kwargs)

    wrapper.__name__ = f.__name__
    return wrapper


# --- 数据导出与分析辅助函数 (保持不变) ---
def export_and_update_excel():
    db = get_db()
    cursor = db.cursor()
    output_dir = "water_data_excel"
    os.makedirs(output_dir, exist_ok=True)

    params_to_export = {
        "悬浮物浓度(TSS)": "result_tss",
        "浊度": "result_turbidity",
    }
    try:
        for param_name, table_name in params_to_export.items():
            cursor.execute(f"SELECT DISTINCT location FROM {table_name}")
            locations = [row['location'] for row in cursor.fetchall()]
            for location in locations:
                filename = f"{location}_{param_name}.xlsx"
                filepath = os.path.join(output_dir, filename)
                last_exported_id = 0
                if os.path.exists(filepath):
                    try:
                        df_existing = pd.read_excel(filepath)
                        if not df_existing.empty:
                            last_exported_id = df_existing['ID'].max()
                    except Exception as e:
                        print(f"读取旧Excel文件 {filename} 失败: {e}。将重新创建。")
                query = f"SELECT id, real_value, inferred_value FROM {table_name} WHERE location = ? AND id > ? ORDER BY id"
                cursor.execute(query, (location, last_exported_id))
                new_data = cursor.fetchall()
                if new_data:
                    df_new = pd.DataFrame(new_data, columns=['ID', '真实值', '反演值'])
                    if os.path.exists(filepath) and last_exported_id > 0:
                        with pd.ExcelWriter(filepath, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                            df_new.to_excel(writer, sheet_name='Sheet1', startrow=writer.sheets['Sheet1'].max_row,
                                            header=False, index=False)
                        print(f"已向 {filename} 追加 {len(df_new)} 条新数据。")
                    else:
                        df_new.to_excel(filepath, index=False)
                        print(f"已创建文件 {filename} 并写入 {len(df_new)} 条数据。")
        return output_dir, True
    except Exception as e:
        print(f"导出Excel时发生错误: {e}")
        return None, False


def analyze_data_from_excel(excel_dir):
    analysis_results = {}
    if not os.path.exists(excel_dir):
        return {}
    for filename in os.listdir(excel_dir):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(excel_dir, filename)
            try:
                df = pd.read_excel(filepath)
                if df.empty:
                    continue
                location, param_name = filename.replace('.xlsx', '').split('_')
                values = df['反演值'].to_numpy()
                stats = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std_dev': np.std(values)
                }
                if location not in analysis_results:
                    analysis_results[location] = {}
                analysis_results[location][param_name] = stats
            except Exception as e:
                print(f"分析Excel文件 {filename} 失败: {e}")
    return analysis_results


# --- API Endpoints (路由函数) ---

@app.route('/register', methods=['POST'])
def register_user():
    # ... (代码不变) ...
    data = request.get_json()
    phone = data.get('phone')
    password = data.get('password')
    if not phone or not password:
        return jsonify({'message': 'Phone and password are required'}), 400
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id FROM user WHERE phone = ?", (phone,))
    if cursor.fetchone():
        return jsonify({'message': 'Phone number already registered'}), 409
    password_hash = pbkdf2_sha256.hash(password)
    try:
        cursor.execute("INSERT INTO user (phone, password_hash, role) VALUES (?, ?, ?)",
                       (phone, password_hash, 'user'))
        db.commit()
    except sqlite3.Error as e:
        db.rollback()
        print(f"Database error during registration: {e}")
        return jsonify({'message': 'Registration failed due to database error'}), 500
    return jsonify({'message': 'User registered successfully'}), 201


@app.route('/login', methods=['POST'])
def login_user():
    # ... (代码不变) ...
    data = request.get_json()
    phone = data.get('phone')
    password = data.get('password')
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, phone, password_hash, role FROM user WHERE phone = ?", (phone,))
    user_row = cursor.fetchone()
    if not user_row:
        return jsonify({'message': 'Invalid credentials'}), 401
    user = dict(user_row)
    if not pbkdf2_sha256.verify(password, user['password_hash']):
        return jsonify({'message': 'Invalid credentials'}), 401
    access_token = generate_token(user['id'], user['phone'], user['role'])
    return jsonify({
        'message': 'Login successful',
        'access_token': access_token,
        'phone': user['phone'],
        'role': user['role']
    }), 200


@app.route('/data', methods=['GET'])
def get_all_location_data():
    # ... (代码不变) ...
    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT id, real_value, inferred_value, location FROM result_tss ORDER BY id")
        tss_data_raw = cursor.fetchall()
        tss_data_map = {item['id']: dict(item) for item in tss_data_raw}
        cursor.execute("SELECT id, real_value, inferred_value, location FROM result_turbidity ORDER BY id")
        turbidity_data_raw = cursor.fetchall()
        turbidity_data_map = {item['id']: dict(item) for item in turbidity_data_raw}
        cursor.execute("SELECT id, real_value, inferred_value FROM result_salinity ORDER BY id")
        salinity_data_raw = cursor.fetchall()
        salinity_data_map = {item['id']: dict(item) for item in salinity_data_raw}
        combined_data = []
        all_unique_ids = set()
        all_unique_ids.update(tss_data_map.keys())
        all_unique_ids.update(turbidity_data_map.keys())
        all_unique_ids.update(salinity_data_map.keys())
        sorted_unique_ids = sorted(list(all_unique_ids))
        if not sorted_unique_ids:
            return jsonify([]), 200
        for point_id in sorted_unique_ids:
            tss_item = tss_data_map.get(point_id)
            turbidity_item = turbidity_data_map.get(point_id)
            salinity_item = salinity_data_map.get(point_id)
            combined_entry = {
                'id': point_id,
                'location': (tss_item.get('location') if tss_item else turbidity_item.get('location')) if (
                            tss_item or turbidity_item) else '未知地点',
                'real_tss': tss_item.get('real_value', None) if tss_item else None,
                'inferred_tss': tss_item.get('inferred_value', None) if tss_item else None,
                'real_salinity': salinity_item.get('real_value', None) if salinity_item else None,
                'inferred_salinity': salinity_item.get('inferred_value', None) if salinity_item else None,
                'real_turbidity': turbidity_item.get('real_value', None) if turbidity_item else None,
                'inferred_turbidity': turbidity_item.get('inferred_value', None) if turbidity_item else None
            }
            combined_data.append(combined_entry)
        return jsonify(combined_data), 200
    except sqlite3.Error as e:
        return jsonify({'message': f'从数据库获取数据失败: {e}'}), 500
    except Exception as e:
        return jsonify({'message': f'发生意外错误: {e}'}), 500


@app.route('/data/<int:data_id>', methods=['PUT', 'DELETE'])
@admin_required
def modify_data(data_id):
    # ... (代码不变) ...
    return jsonify({'message': '当前多表数据库设计不支持通过单个ID进行更新或删除操作。'}), 400


# ==================== 【核心修改区域】 ====================
@app.route('/chat_with_ai', methods=['POST'])
def chat_with_ai():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'Message is required'}), 400

    # 1. 导出/更新 Excel 文件
    excel_dir, success = export_and_update_excel()
    if not success:
        return jsonify({'ai_response': 'AI 助手出错: 准备数据文件时失败，请检查后端日志。'}), 500

    # 2. 从 Excel 文件分析数据
    analysis = analyze_data_from_excel(excel_dir)

    # 3. 动态生成给AI的上下文，聚焦于本次采样
    data_context = "背景：我们对七堡和盐官两个不同水域的多个采样点进行了**一次性水质采样**，并通过AI模型对悬浮物(TSS)和浊度进行了反演。以下是本次采样数据的统计分析摘要：\n\n"
    if not analysis:
        data_context += "未能从数据文件中加载分析结果。\n"
    else:
        for location, params in analysis.items():
            data_context += f"===== 地点: {location} =====\n"
            for param_name, stats in params.items():
                data_context += f"--- 参数: {param_name} ---\n"
                data_context += f"  - 采样点数量: {stats['count']}\n"
                data_context += f"  - 平均值: {stats['mean']:.2f}\n"
                data_context += f"  - 数据范围 (最小值-最大值): {stats['min']:.2f} - {stats['max']:.2f}\n"
                data_context += f"  - 数据离散程度 (标准差): {stats['std_dev']:.2f} (值越小代表该地点内水质越均匀)\n"
            data_context += "\n"

    # 4. 构建全新的、针对单次采样的 "富提示词"
    system_prompt = (
        "你是一位专业的水环境科学分析师'小然'。你的任务是解读我提供的**单次采样**的水质数据统计摘要，并结合你的专业知识，生成一份深入的分析报告。\n\n"
        "**你的分析必须聚焦于以下几个方面：**\n"
        "1.  **解读数据分布**：根据每个地点的**平均值**、**数据范围**和**离散程度**，评价该区域内水质的总体水平和均匀性。例如，'七堡的悬浮物浓度均值较高，且数据范围宽、离散程度大，表明该区域内水质不均匀，可能存在点状污染源或强水动力影响'。\n"
        "2.  **与水质标准对比（关键）**：将分析出的数值与公认的地表水环境质量标准（如GB 3838-2002）进行对比。请明确指出哪些指标在哪个地点可能处于超标或接近临界值的状态。例如，'一般认为饮用水源地的浊度不宜高于1 NTU，景观水体不宜高于20 NTU。基于此，请评价当前浊度水平'。\n"
        "3.  **参数关联性分析**：结合专业知识，分析**悬浮物(TSS)和浊度**之间的关系。通常它们高度正相关，请评价本次采样数据是否符合这一规律，并解释其环境意义。\n"
        "4.  **风险评估与结论**：综合以上所有分析，对七堡和盐官两个地点的水质状况进行最终评价，指出各自面临的主要水质问题（例如，是整体浑浊，还是局部污染？），并提出有针对性的下一步调查建议。\n\n"
        "请使用专业、流畅的中文撰写报告，并用Markdown格式化以突出重点。"
    )

    full_user_prompt = f"""
【本次采样数据分析摘要】
{data_context}

【用户具体要求】
{user_message}

请根据以上摘要和我的要求，并严格遵循你作为水环境科学分析师的角色和分析框架，开始撰写你的专业报告。
"""

    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {DEEPSEEK_API_KEY}'}
    payload = {
        'model': 'deepseek-chat',
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': full_user_prompt}
        ],
        'stream': False
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        ai_response_data = response.json()
        ai_text = ai_response_data.get('choices', [{}])[0].get('message', {}).get('content', 'AI未能生成有效回复。')
        return jsonify({'ai_response': ai_text}), 200
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with DeepSeek API: {e}")
        return jsonify({'ai_response': f'AI 助手出错: 与模型通信失败，错误详情: {e}'}), 503
    except Exception as e:
        print(f"An unexpected error occurred in chat_with_ai: {e}")
        return jsonify({'ai_response': f'AI 助手出错: 服务器内部发生未知错误。'}), 500


# --- 主程序入口 ---
if __name__ == '__main__':
    # 在应用启动时运行数据库初始化
    with app.app_context():
        init_db()
    app.run(debug=True)