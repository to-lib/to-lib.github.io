---
sidebar_position: 19
---

# WebSocket 实时通信

> [!TIP]
> **WebSocket 的优势**: WebSocket 提供全双工通信，服务器可以主动向客户端推送数据，非常适合实时聊天、实时通知、在线协作等场景。

## WebSocket vs HTTP

| 特性 | HTTP | WebSocket |
|------|------|-----------|
| 通信方式 | 单向（请求-响应） | 双向（全双工） |
| 连接 | 短连接 | 长连接 |
| 服务器推送 | 不支持（需要轮询） | 支持 |
| 开销 | 每次请求都有HTTP头 | 只有首次握手有HTTP头 |
| 实时性 | 差（需要轮询） | 好 |

## 添加依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-websocket</artifactId>
</dependency>
```

## WebSocket 配置

```java
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.*;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {
    
    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(new ChatWebSocketHandler(), "/ws/chat")
            .setAllowedOrigins("*"); // 允许所有域访问
    }
}
```

## WebSocket Handler

```java
import org.springframework.web.socket.*;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class ChatWebSocketHandler extends TextWebSocketHandler {
    
    private static final Set<WebSocketSession> sessions = new CopyOnWriteArraySet<>();
    
    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        sessions.add(session);
        log.info("新连接建立: {}, 当前在线: {}", session.getId(), sessions.size());
    }
    
    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        String payload = message.getPayload();
        log.info("收到消息: {} from {}", payload, session.getId());
        
        // 广播消息给所有连接
        for (WebSocketSession s : sessions) {
            if (s.isOpen()) {
                s.sendMessage(new TextMessage("用户" + session.getId() + ": " + payload));
            }
        }
    }
    
    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        sessions.remove(session);
        log.info("连接关闭: {}, 当前在线: {}", session.getId(), sessions.size());
    }
    
    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) throws Exception {
        log.error("传输错误: {}", session.getId(), exception);
        sessions.remove(session);
    }
}
```

## STOMP 协议（推荐）

STOMP (Simple Text Oriented Messaging Protocol) 是一个简单的消息协议，Spring Boot 提供了很好的支持。

### 配置 STOMP

```java
import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.web.socket.config.annotation.*;

@Configuration
@EnableWebSocketMessageBroker
public class WebSocketStompConfig implements WebSocketMessageBrokerConfigurer {
    
    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        // 注册 STOMP 端点
        registry.addEndpoint("/ws")
            .setAllowedOriginPatterns("*")
            .withSockJS(); // 启用 SockJS 降级支持
    }
    
    @Override
    public void configureMessageBroker(MessageBrokerRegistry registry) {
        // 配置消息代理
        registry.enableSimpleBroker("/topic", "/queue"); // 订阅前缀
        registry.setApplicationDestinationPrefixes("/app"); // 发送前缀
        registry.setUserDestinationPrefix("/user"); // 点对点消息前缀
    }
}
```

### 消息控制器

```java
import org.springframework.messaging.handler.annotation.*;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Controller;

@Controller
@Slf4j
public class ChatController {
    
    @Autowired
    private SimpMessagingTemplate messagingTemplate;
    
    // 接收客户端消息并广播
    @MessageMapping("/chat.send") // 客户端发送到 /app/chat.send
    @SendTo("/topic/public") // 广播到 /topic/public
    public ChatMessage sendMessage(@Payload ChatMessage message) {
        log.info("收到消息: {}", message);
        return message;
    }
    
    // 用户加入
    @MessageMapping("/chat.join")
    @SendTo("/topic/public")
    public ChatMessage joinChat(@Payload ChatMessage message,
                                SimpMessageHeaderAccessor headerAccessor) {
        headerAccessor.getSessionAttributes().put("username", message.getSender());
        message.setType(ChatMessage.MessageType.JOIN);
        message.setContent(message.getSender() + " 加入了聊天室");
        return message;
    }
    
    // 点对点消息
    @MessageMapping("/chat.private")
    public void sendPrivateMessage(@Payload PrivateMessage message) {
        log.info("发送私信: from {} to {}", message.getSender(), message.getRecipient());
        
        messagingTemplate.convertAndSendToUser(
            message.getRecipient(),
            "/queue/private",
            message
        );
    }
}

// 消息实体
@Data
@AllArgsConstructor
@NoArgsConstructor
public class ChatMessage {
    private MessageType type;
    private String content;
    private String sender;
    
    public enum MessageType {
        CHAT, JOIN, LEAVE
    }
}

@Data
public class PrivateMessage {
    private String sender;
    private String recipient;
    private String content;
}
```

### 前端代码（JavaScript）

```html
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Chat</title>
    <script src="https://cdn.jsdelivr.net/npm/sockjs-client@1/dist/sockjs.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/stompjs@2.3.3/lib/stomp.min.js"></script>
</head>
<body>
    <div id="chat">
        <div id="messages"></div>
        <input type="text" id="messageInput" placeholder="输入消息">
        <button onclick="sendMessage()">发送</button>
    </div>

    <script>
        let stompClient = null;
        let username = 'User' + Math.floor(Math.random() * 1000);

        function connect() {
            const socket = new SockJS('/ws');
            stompClient = Stomp.over(socket);
            
            stompClient.connect({}, function(frame) {
                console.log('Connected: ' + frame);
                
                // 订阅公共频道
                stompClient.subscribe('/topic/public', function(message) {
                    showMessage(JSON.parse(message.body));
                });
                
                // 订阅私信
                stompClient.subscribe('/user/queue/private', function(message) {
                    showPrivateMessage(JSON.parse(message.body));
                });
                
                // 发送加入消息
                stompClient.send("/app/chat.join", {}, JSON.stringify({
                    sender: username,
                    type: 'JOIN'
                }));
            });
        }

        function sendMessage() {
            const messageInput = document.getElementById('messageInput');
            const message = {
                sender: username,
                content: messageInput.value,
                type: 'CHAT'
            };
            
            stompClient.send("/app/chat.send", {}, JSON.stringify(message));
            messageInput.value = '';
        }

        function showMessage(message) {
            const messagesDiv = document.getElementById('messages');
            const messageElement = document.createElement('div');
            messageElement.textContent = message.sender + ': ' + message.content;
            messagesDiv.appendChild(messageElement);
        }

        // 连接到 WebSocket
        connect();
    </script>
</body>
</html>
```

## 实战：在线聊天室

完整的聊天室实现，包括用户管理、消息历史等。

```java
@Service
@Slf4j
public class ChatService {
    
    private final Map<String, User> onlineUsers = new ConcurrentHashMap<>();
    private final List<ChatMessage> messageHistory = new CopyOnWriteArrayList<>();
    
    @Autowired
    private SimpMessagingTemplate messagingTemplate;
    
    public void userJoin(String sessionId, User user) {
        onlineUsers.put(sessionId, user);
        
        // 广播用户列表更新
        messagingTemplate.convertAndSend("/topic/users", getOnlineUsers());
        
        // 发送历史消息给新用户
        messagingTemplate.convertAndSendToUser(
            sessionId,
            "/queue/history",
            messageHistory
        );
    }
    
    public void userLeave(String sessionId) {
        User user = onlineUsers.remove(sessionId);
        if (user != null) {
            ChatMessage leaveMessage = new ChatMessage();
            leaveMessage.setType(ChatMessage.MessageType.LEAVE);
            leaveMessage.setSender(user.getUsername());
            leaveMessage.setContent(user.getUsername() + " 离开了聊天室");
            
            broadcastMessage(leaveMessage);
            messagingTemplate.convertAndSend("/topic/users", getOnlineUsers());
        }
    }
    
    public void broadcastMessage(ChatMessage message) {
        message.setTimestamp(LocalDateTime.now());
        messageHistory.add(message);
        
        // 只保留最近100条消息
        if (messageHistory.size() > 100) {
            messageHistory.remove(0);
        }
        
        messagingTemplate.convertAndSend("/topic/public", message);
    }
    
    public List<User> getOnlineUsers() {
        return new ArrayList<>(onlineUsers.values());
    }
}
```

## 安全性

### WebSocket 认证

```java
@Configuration
public class WebSocketSecurityConfig {
    
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/ws/**").authenticated()
                .anyRequest().permitAll()
            )
            .csrf(csrf -> csrf.disable());
        return http.build();
    }
}

// 在 WebSocket 配置中添加拦截器
@Configuration
@EnableWebSocketMessageBroker
public class WebSocketStompConfig implements WebSocketMessageBrokerConfigurer {
    
    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/ws")
            .setAllowedOriginPatterns("*")
            .addInterceptors(new HttpSessionHandshakeInterceptor())
            .withSockJS();
    }
    
    @Override
    public void configureClientInboundChannel(ChannelRegistration registration) {
        registration.interceptors(new AuthChannelInterceptor());
    }
}

// 认证拦截器
@Component
public class AuthChannelInterceptor implements ChannelInterceptor {
    
    @Override
    public Message<?> preSend(Message<?> message, MessageChannel channel) {
        StompHeaderAccessor accessor = 
            MessageHeaderAccessor.getAccessor(message, StompHeaderAccessor.class);
        
        if (StompCommand.CONNECT.equals(accessor.getCommand())) {
            String token = accessor.getFirstNativeHeader("Authorization");
            
            if (token == null || !validateToken(token)) {
                throw new IllegalArgumentException("Invalid token");
            }
            
            // 设置用户信息
            accessor.setUser(new UsernamePasswordAuthenticationToken(
                getUserFromToken(token), null, Collections.emptyList()
            ));
        }
        
        return message;
    }
    
    private boolean validateToken(String token) {
        // 验证 token
        return true;
    }
    
    private String getUserFromToken(String token) {
        // 从 token 获取用户名
        return "user";
    }
}
```

## 最佳实践

> [!TIP]
> **WebSocket 最佳实践**：
>
> 1. **使用 STOMP** - 比原生 WebSocket 更简单易用
> 2. **启用 SockJS** - 提供降级支持，确保兼容性
> 3. **心跳检测** - 定期发送心跳消息，检测连接状态
> 4. **断线重连** - 客户端实现自动重连机制
> 5. **消息确认** - 关键消息需要确认机制
> 6. **限流** - 防止消息轰炸，限制消息频率
> 7. **监控** - 监控连接数、消息量等指标

## 总结

- **WebSocket** - 全双工实时通信协议
- **STOMP** - 简化的消息协议，推荐使用
- **SockJS** - WebSocket 降级方案
- **实时场景** - 聊天、通知、协作、游戏等

下一步学习 [Docker 容器化](/docs/springboot/docker)。
