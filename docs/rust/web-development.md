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

## Tower 服务栈

Tower 是 Rust 异步服务抽象的核心，Axum 完全基于 Tower 构建。

### Tower Service Trait

```rust
use std::task::{Context, Poll};
use std::future::Future;
use std::pin::Pin;

// Tower Service trait 的核心
pub trait Service<Request> {
    type Response;
    type Error;
    type Future: Future<Output = Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>>;
    fn call(&mut self, req: Request) -> Self::Future;
}
```

### 常用 Tower 中间件

```rust
use axum::{Router, routing::get};
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    timeout::TimeoutLayer,
    compression::CompressionLayer,
    limit::RequestBodyLimitLayer,
    request_id::{MakeRequestId, RequestId, SetRequestIdLayer, PropagateRequestIdLayer},
};
use std::time::Duration;

#[derive(Clone)]
struct MyMakeRequestId;

impl MakeRequestId for MyMakeRequestId {
    fn make_request_id<B>(&mut self, _: &http::Request<B>) -> Option<RequestId> {
        Some(RequestId::new(uuid::Uuid::new_v4().to_string().parse().unwrap()))
    }
}

fn create_app() -> Router {
    Router::new()
        .route("/", get(|| async { "Hello" }))
        .layer(
            ServiceBuilder::new()
                // 从内到外执行
                .layer(SetRequestIdLayer::x_request_id(MyMakeRequestId))
                .layer(PropagateRequestIdLayer::x_request_id())
                .layer(TraceLayer::new_for_http())
                .layer(CompressionLayer::new())
                .layer(TimeoutLayer::new(Duration::from_secs(30)))
                .layer(RequestBodyLimitLayer::new(1024 * 1024))  // 1MB
                .layer(CorsLayer::permissive())
        )
}
```

### 自定义 Tower Layer

```rust
use axum::{body::Body, http::Request, response::Response};
use std::task::{Context, Poll};
use tower::{Layer, Service};
use std::future::Future;
use std::pin::Pin;

#[derive(Clone)]
pub struct TimingLayer;

impl<S> Layer<S> for TimingLayer {
    type Service = TimingService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TimingService { inner }
    }
}

#[derive(Clone)]
pub struct TimingService<S> {
    inner: S,
}

impl<S> Service<Request<Body>> for TimingService<S>
where
    S: Service<Request<Body>, Response = Response> + Clone + Send + 'static,
    S::Future: Send,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let start = std::time::Instant::now();
        let method = req.method().clone();
        let uri = req.uri().clone();
        let future = self.inner.call(req);

        Box::pin(async move {
            let response = future.await;
            let elapsed = start.elapsed();
            tracing::info!(%method, %uri, ?elapsed, "请求完成");
            response
        })
    }
}
```

## OAuth2 认证

### OAuth2 授权码流程

```rust
use axum::{
    extract::{Query, State},
    response::Redirect,
    routing::get,
    Router,
};
use oauth2::{
    AuthorizationCode, AuthUrl, ClientId, ClientSecret, CsrfToken,
    PkceCodeChallenge, RedirectUrl, Scope, TokenResponse, TokenUrl,
    basic::BasicClient,
};
use serde::Deserialize;

#[derive(Clone)]
struct AppState {
    oauth_client: BasicClient,
}

fn create_oauth_client() -> BasicClient {
    BasicClient::new(
        ClientId::new("your-client-id".to_string()),
        Some(ClientSecret::new("your-client-secret".to_string())),
        AuthUrl::new("https://provider.com/oauth/authorize".to_string()).unwrap(),
        Some(TokenUrl::new("https://provider.com/oauth/token".to_string()).unwrap()),
    )
    .set_redirect_uri(RedirectUrl::new("http://localhost:3000/callback".to_string()).unwrap())
}

// 步骤 1: 重定向到 OAuth 提供商
async fn login(State(state): State<AppState>) -> Redirect {
    let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();

    let (auth_url, csrf_token) = state.oauth_client
        .authorize_url(CsrfToken::new_random)
        .add_scope(Scope::new("read:user".to_string()))
        .set_pkce_challenge(pkce_challenge)
        .url();

    // 存储 csrf_token 和 pkce_verifier（实际应用中用 session/redis）

    Redirect::to(auth_url.as_str())
}

#[derive(Deserialize)]
struct CallbackParams {
    code: String,
    state: String,
}

// 步骤 2: 处理回调，交换 token
async fn callback(
    State(state): State<AppState>,
    Query(params): Query<CallbackParams>,
) -> Result<String, String> {
    // 验证 CSRF token
    // 获取之前存储的 pkce_verifier

    let token_result = state.oauth_client
        .exchange_code(AuthorizationCode::new(params.code))
        // .set_pkce_verifier(pkce_verifier)
        .request_async(oauth2::reqwest::async_http_client)
        .await
        .map_err(|e| format!("Token exchange failed: {}", e))?;

    let access_token = token_result.access_token().secret();
    Ok(format!("登录成功！Token: {}...", &access_token[..20]))
}
```

## SeaORM 数据库操作

```toml
[dependencies]
sea-orm = { version = "0.12", features = ["sqlx-postgres", "runtime-tokio-native-tls"] }
```

### 定义实体

```rust
use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "users")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i64,
    pub username: String,
    pub email: String,
    pub created_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_many = "super::post::Entity")]
    Posts,
}

impl Related<super::post::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Posts.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
```

### CRUD 操作

```rust
use sea_orm::{Database, DatabaseConnection, EntityTrait, ActiveModelTrait, Set, QueryFilter, ColumnTrait};

async fn crud_examples(db: &DatabaseConnection) -> Result<(), sea_orm::DbErr> {
    // 创建
    let user = user::ActiveModel {
        username: Set("alice".to_string()),
        email: Set("alice@example.com".to_string()),
        ..Default::default()
    };
    let user = user.insert(db).await?;

    // 查询
    let found = User::find_by_id(user.id).one(db).await?;
    let all_users = User::find().all(db).await?;
    let filtered = User::find()
        .filter(user::Column::Username.contains("ali"))
        .all(db).await?;

    // 更新
    let mut active: user::ActiveModel = found.unwrap().into();
    active.email = Set("newemail@example.com".to_string());
    active.update(db).await?;

    // 删除
    User::delete_by_id(user.id).exec(db).await?;

    Ok(())
}
```

## 高并发优化

### 连接池配置

```rust
use sqlx::postgres::PgPoolOptions;
use deadpool_redis::{Config as RedisConfig, Runtime};

async fn setup_pools() {
    // PostgreSQL 连接池
    let pg_pool = PgPoolOptions::new()
        .max_connections(100)
        .min_connections(10)
        .acquire_timeout(std::time::Duration::from_secs(3))
        .idle_timeout(std::time::Duration::from_secs(600))
        .connect("postgres://user:pass@localhost/db")
        .await
        .unwrap();

    // Redis 连接池
    let redis_cfg = RedisConfig::from_url("redis://localhost:6379");
    let redis_pool = redis_cfg.create_pool(Some(Runtime::Tokio1)).unwrap();
}
```

### 请求限流

```rust
use axum::{
    extract::State,
    http::StatusCode,
    middleware::{self, Next},
    response::Response,
};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Clone)]
struct RateLimiter {
    semaphore: Arc<Semaphore>,
}

impl RateLimiter {
    fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }
}

async fn rate_limit_middleware(
    State(limiter): State<RateLimiter>,
    request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let _permit = limiter.semaphore
        .try_acquire()
        .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;

    Ok(next.run(request).await)
}
```

### 缓存策略

```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};

struct CacheEntry<T> {
    value: T,
    expires_at: Instant,
}

struct Cache<T> {
    data: RwLock<HashMap<String, CacheEntry<T>>>,
    ttl: Duration,
}

impl<T: Clone> Cache<T> {
    fn new(ttl: Duration) -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
            ttl,
        }
    }

    async fn get(&self, key: &str) -> Option<T> {
        let data = self.data.read().await;
        data.get(key)
            .filter(|entry| entry.expires_at > Instant::now())
            .map(|entry| entry.value.clone())
    }

    async fn set(&self, key: String, value: T) {
        let mut data = self.data.write().await;
        data.insert(key, CacheEntry {
            value,
            expires_at: Instant::now() + self.ttl,
        });
    }
}
```

### 异步批处理

```rust
use tokio::sync::mpsc;
use std::time::Duration;

struct BatchProcessor<T> {
    sender: mpsc::Sender<T>,
}

impl<T: Send + 'static> BatchProcessor<T> {
    fn new<F>(batch_size: usize, timeout: Duration, handler: F) -> Self
    where
        F: Fn(Vec<T>) + Send + 'static,
    {
        let (tx, mut rx) = mpsc::channel::<T>(1000);

        tokio::spawn(async move {
            let mut batch = Vec::with_capacity(batch_size);
            let mut interval = tokio::time::interval(timeout);

            loop {
                tokio::select! {
                    Some(item) = rx.recv() => {
                        batch.push(item);
                        if batch.len() >= batch_size {
                            handler(std::mem::take(&mut batch));
                        }
                    }
                    _ = interval.tick() => {
                        if !batch.is_empty() {
                            handler(std::mem::take(&mut batch));
                        }
                    }
                }
            }
        });

        Self { sender: tx }
    }

    async fn submit(&self, item: T) -> Result<(), mpsc::error::SendError<T>> {
        self.sender.send(item).await
    }
}
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
