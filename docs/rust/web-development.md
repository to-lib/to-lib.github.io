---
sidebar_position: 14
title: Web 后端开发
---

# Web 后端开发

本文介绍使用 Rust 构建 Web 后端服务，重点讲解 Axum 框架及其生态系统。

## Axum 简介

Axum 是由 Tokio 团队开发的 Web 框架，特点：

- **无宏路由**：类型安全的路由定义
- **Tower 生态**：复用丰富的中间件
- **提取器模式**：声明式请求解析

```toml
[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
tower-http = { version = "0.5", features = ["cors", "trace"] }
```

## 快速开始

```rust
use axum::{routing::get, Router};

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(|| async { "Hello, World!" }));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

## 路由

```rust
use axum::{routing::{get, post}, Router};

// 基础路由
let app = Router::new()
    .route("/users", get(list_users).post(create_user))
    .route("/users/:id", get(get_user));

// 路由嵌套
let api_routes = Router::new()
    .nest("/users", user_routes)
    .nest("/posts", post_routes);

let app = Router::new().nest("/api/v1", api_routes);
```

## 提取器

```rust
use axum::extract::{Path, Query, State, Json};
use serde::Deserialize;

// 路径参数
async fn get_user(Path(id): Path<u64>) -> String {
    format!("User {}", id)
}

// 查询参数
#[derive(Deserialize)]
struct Pagination {
    page: Option<u32>,
}

async fn list_items(Query(pagination): Query<Pagination>) -> String {
    format!("Page {}", pagination.page.unwrap_or(1))
}

// JSON 请求体
async fn create_item(Json(payload): Json<CreateItem>) -> Json<Item> {
    // ...
}
```

### 自定义提取器

```rust
use axum::{async_trait, extract::FromRequestParts, http::{request::Parts, StatusCode}};

struct AuthUser { user_id: u64 }

#[async_trait]
impl<S: Send + Sync> FromRequestParts<S> for AuthUser {
    type Rejection = StatusCode;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Self::Rejection> {
        let header = parts.headers.get("Authorization")
            .and_then(|v| v.to_str().ok())
            .ok_or(StatusCode::UNAUTHORIZED)?;

        let user_id = validate_token(header).map_err(|_| StatusCode::UNAUTHORIZED)?;
        Ok(AuthUser { user_id })
    }
}
```

## 中间件

```rust
use tower_http::{cors::CorsLayer, trace::TraceLayer, timeout::TimeoutLayer};
use std::time::Duration;

let app = Router::new()
    .route("/", get(handler))
    .layer(TraceLayer::new_for_http())
    .layer(CorsLayer::permissive())
    .layer(TimeoutLayer::new(Duration::from_secs(30)));
```

### 自定义中间件

```rust
use axum::{middleware::{self, Next}, extract::Request, response::Response};

async fn logging_middleware(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start = std::time::Instant::now();

    let response = next.run(request).await;

    tracing::info!(%method, %uri, status = %response.status(), duration = ?start.elapsed());
    response
}

let app = Router::new()
    .layer(middleware::from_fn(logging_middleware));
```

## SQLx 数据库

```toml
[dependencies]
sqlx = { version = "0.7", features = ["runtime-tokio", "postgres"] }
```

```rust
use sqlx::{PgPool, FromRow};

#[derive(FromRow, Serialize)]
struct User { id: i64, name: String, email: String }

async fn list_users(State(pool): State<PgPool>) -> Result<Json<Vec<User>>, StatusCode> {
    let users = sqlx::query_as::<_, User>("SELECT * FROM users")
        .fetch_all(&pool)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(users))
}

async fn create_user(State(pool): State<PgPool>, Json(payload): Json<CreateUser>) -> Result<Json<User>, StatusCode> {
    let user = sqlx::query_as::<_, User>(
        "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *"
    )
    .bind(&payload.name)
    .bind(&payload.email)
    .fetch_one(&pool)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(user))
}
```

## Redis 集成

```rust
use deadpool_redis::{Config, Runtime, Pool};
use redis::AsyncCommands;

async fn cache_user(State(pool): State<Pool>, Path(id): Path<u64>, Json(user): Json<User>) -> Result<StatusCode, StatusCode> {
    let mut conn = pool.get().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let key = format!("user:{}", id);
    let value = serde_json::to_string(&user).unwrap();
    conn.set_ex::<_, _, ()>(&key, &value, 3600).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(StatusCode::OK)
}
```

## JWT 认证

```rust
use jsonwebtoken::{encode, decode, Header, EncodingKey, DecodingKey, Validation};

#[derive(Serialize, Deserialize)]
struct Claims { sub: String, exp: usize }

fn create_token(user_id: &str) -> Result<String, jsonwebtoken::errors::Error> {
    let claims = Claims {
        sub: user_id.to_string(),
        exp: (chrono::Utc::now() + chrono::Duration::hours(24)).timestamp() as usize,
    };
    encode(&Header::default(), &claims, &EncodingKey::from_secret(b"secret"))
}

fn verify_token(token: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
    let data = decode::<Claims>(token, &DecodingKey::from_secret(b"secret"), &Validation::default())?;
    Ok(data.claims)
}
```

## 错误处理

```rust
use axum::{response::{IntoResponse, Response}, http::StatusCode, Json};

enum AppError {
    NotFound,
    BadRequest(String),
    InternalError(anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::NotFound => (StatusCode::NOT_FOUND, "Not found"),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.as_str()),
            AppError::InternalError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Internal error"),
        };
        (status, Json(serde_json::json!({"error": message}))).into_response()
    }
}
```

## 项目结构

```
web-api/
├── src/
│   ├── main.rs
│   ├── config.rs
│   ├── error.rs
│   ├── routes/
│   │   ├── mod.rs
│   │   └── users.rs
│   ├── middleware/
│   ├── models/
│   └── db/
├── migrations/
└── tests/
```

## 最佳实践

| 方面   | 建议                             |
| ------ | -------------------------------- |
| 架构   | routes → services → repositories |
| 状态   | `Arc<RwLock<T>>` 或连接池        |
| 错误   | 自定义错误实现 `IntoResponse`    |
| 认证   | 自定义提取器 + 中间件            |
| 数据库 | SQLx 编译时检查                  |
| 日志   | tracing + TraceLayer             |
