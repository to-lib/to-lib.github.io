---
sidebar_position: 16
title: 地理位置
---

# Redis 地理位置功能

Redis 3.2 引入了 GEO（地理位置）功能，支持存储地理坐标，进行距离计算和范围查询。

## GEO 简介

### 核心能力

- **存储位置** - 存储经纬度坐标
- **距离计算** - 计算两点之间的距离
- **范围查询** - 查询指定范围内的位置
- **附近搜索** - 查找附近的点

### 底层实现

GEO 使用 Sorted Set 存储数据：

- 经纬度使用 GeoHash 编码为 52 位整数
- 作为分数存储在 Sorted Set 中
- 支持标准 Sorted Set 命令

## 基本命令

### GEOADD - 添加位置

```bash
# 语法
GEOADD key longitude latitude member [longitude latitude member ...]

# 添加单个位置
GEOADD restaurants 116.403963 39.915119 "天安门"

# 批量添加
GEOADD restaurants 116.469392 39.95155 "三里屯" 116.27641 39.99841 "颐和园"
```

**参数说明**：

- `longitude` - 经度（-180 到 180）
- `latitude` - 纬度（-85.05112878 到 85.05112878）
- `member` - 位置名称

### GEOPOS - 获取坐标

```bash
# 获取一个或多个位置的坐标
GEOPOS restaurants "天安门" "三里屯"

# 返回
1) 1) "116.40396267175674438"
   2) "39.91511970338637383"
2) 1) "116.46939247846603394"
   2) "39.95154958985968577"
```

### GEODIST - 计算距离

```bash
# 语法
GEODIST key member1 member2 [unit]

# 计算天安门到三里屯的距离
GEODIST restaurants "天安门" "三里屯" km
# "7.8364"

# 单位选项
# m - 米（默认）
# km - 千米
# mi - 英里
# ft - 英尺
```

### GEOHASH - 获取 GeoHash

```bash
# 获取 GeoHash 值
GEOHASH restaurants "天安门"
# "wx4g0b7xr70"
```

GeoHash 特点：

- 11 位字符串
- 相似位置前缀相同
- 可用于 URL 或数据库索引

### GEORADIUS - 范围查询

```bash
# 语法
GEORADIUS key longitude latitude radius unit [WITHCOORD] [WITHDIST] [WITHHASH] [COUNT count] [ASC|DESC]

# 查询天安门5公里范围内的位置
GEORADIUS restaurants 116.403963 39.915119 5 km WITHDIST COUNT 10

# 返回距离和坐标
GEORADIUS restaurants 116.403963 39.915119 5 km WITHDIST WITHCOORD COUNT 10
```

参数说明：

- `WITHCOORD` - 返回坐标
- `WITHDIST` - 返回距离
- `WITHHASH` - 返回 GeoHash
- `COUNT` - 限制返回数量
- `ASC` / `DESC` - 按距离排序

### GEORADIUSBYMEMBER - 以成员为中心查询

```bash
# 查询天安门附近5公里的位置
GEORADIUSBYMEMBER restaurants "天安门" 5 km WITHDIST ASC COUNT 10
```

### GEOSEARCH - 范围搜索（Redis 6.2+）

```bash
# 按圆形范围搜索
GEOSEARCH restaurants FROMMEMBER "天安门" BYRADIUS 5 km ASC COUNT 10

# 按矩形范围搜索
GEOSEARCH restaurants FROMMEMBER "天安门" BYBOX 10 10 km ASC COUNT 10

# 从指定坐标搜索
GEOSEARCH restaurants FROMLONLAT 116.403963 39.915119 BYRADIUS 5 km ASC
```

### GEOSEARCHSTORE - 存储搜索结果

```bash
# 将搜索结果存储到新键
GEOSEARCHSTORE nearby_restaurants restaurants FROMMEMBER "天安门" BYRADIUS 5 km ASC COUNT 10
```

## 实战案例

### 1. 附近的人

```java
public class NearbyPeople {
    private Jedis jedis;
    private static final String KEY = "people:locations";

    // 更新用户位置
    public void updateLocation(String userId, double longitude, double latitude) {
        jedis.geoadd(KEY, longitude, latitude, userId);
        // 设置过期时间（可选，需要单独的键）
    }

    // 查找附近的人
    public List<GeoRadiusResponse> findNearby(double longitude, double latitude,
                                               double radius, GeoUnit unit, int count) {
        return jedis.georadius(KEY, longitude, latitude, radius, unit,
            GeoRadiusParam.geoRadiusParam()
                .withDist()
                .withCoord()
                .sortAscending()
                .count(count)
        );
    }

    // 查找某用户附近的人
    public List<GeoRadiusResponse> findNearbyByMember(String userId,
                                                       double radius, GeoUnit unit, int count) {
        return jedis.georadiusByMember(KEY, userId, radius, unit,
            GeoRadiusParam.geoRadiusParam()
                .withDist()
                .withCoord()
                .sortAscending()
                .count(count)
        );
    }

    // 删除用户位置
    public void removeLocation(String userId) {
        jedis.zrem(KEY, userId);  // GEO 底层是 ZSET
    }
}

// 使用示例
NearbyPeople nearby = new NearbyPeople(jedis);

// 更新位置
nearby.updateLocation("user:1001", 116.403963, 39.915119);
nearby.updateLocation("user:1002", 116.469392, 39.95155);

// 查找 5 公里内的人
List<GeoRadiusResponse> people = nearby.findNearby(
    116.403963, 39.915119, 5, GeoUnit.KM, 10);

for (GeoRadiusResponse person : people) {
    System.out.println(person.getMemberByString() +
        " - 距离: " + person.getDistance() + " km");
}
```

### 2. 附近的店铺

```java
public class NearbyShops {
    private Jedis jedis;
    private static final String KEY = "shops:locations";

    // 添加店铺
    public void addShop(String shopId, double longitude, double latitude,
                        Map<String, String> shopInfo) {
        // 添加位置
        jedis.geoadd(KEY, longitude, latitude, shopId);

        // 存储店铺详情
        jedis.hmset("shop:" + shopId, shopInfo);
    }

    // 搜索附近店铺
    public List<Map<String, Object>> searchNearby(double longitude, double latitude,
                                                   double radius, int count) {
        List<GeoRadiusResponse> locations = jedis.georadius(
            KEY, longitude, latitude, radius, GeoUnit.M,
            GeoRadiusParam.geoRadiusParam()
                .withDist()
                .sortAscending()
                .count(count)
        );

        List<Map<String, Object>> result = new ArrayList<>();
        for (GeoRadiusResponse loc : locations) {
            String shopId = loc.getMemberByString();

            Map<String, Object> shop = new HashMap<>();
            shop.put("id", shopId);
            shop.put("distance", loc.getDistance());
            shop.put("info", jedis.hgetAll("shop:" + shopId));

            result.add(shop);
        }
        return result;
    }

    // 按分类搜索
    public List<Map<String, Object>> searchByCategory(double longitude, double latitude,
                                                       double radius, String category) {
        String key = "shops:" + category + ":locations";

        List<GeoRadiusResponse> locations = jedis.georadius(
            key, longitude, latitude, radius, GeoUnit.M,
            GeoRadiusParam.geoRadiusParam()
                .withDist()
                .sortAscending()
                .count(50)
        );

        // 同上处理...
        return processLocations(locations);
    }
}
```

### 3. 地理围栏

检测用户是否在指定区域内：

```java
public class GeoFence {
    private Jedis jedis;

    // 创建地理围栏（以某点为中心的圆形区域）
    public void createFence(String fenceId, double longitude, double latitude,
                            double radius) {
        Map<String, String> fence = new HashMap<>();
        fence.put("longitude", String.valueOf(longitude));
        fence.put("latitude", String.valueOf(latitude));
        fence.put("radius", String.valueOf(radius));
        jedis.hmset("fence:" + fenceId, fence);
    }

    // 检测用户是否在围栏内
    public boolean isInFence(String fenceId, double userLon, double userLat) {
        Map<String, String> fence = jedis.hgetAll("fence:" + fenceId);

        double fenceLon = Double.parseDouble(fence.get("longitude"));
        double fenceLat = Double.parseDouble(fence.get("latitude"));
        double radius = Double.parseDouble(fence.get("radius"));

        // 使用临时键计算距离
        String tempKey = "temp:geofence:" + System.currentTimeMillis();
        jedis.geoadd(tempKey, fenceLon, fenceLat, "center");
        jedis.geoadd(tempKey, userLon, userLat, "user");

        Double distance = jedis.geodist(tempKey, "center", "user", GeoUnit.M);
        jedis.del(tempKey);

        return distance != null && distance <= radius;
    }

    // 进入/离开围栏事件
    public void checkFenceEvent(String userId, String fenceId,
                                 double userLon, double userLat) {
        String statusKey = "fence:status:" + fenceId + ":" + userId;
        boolean currentlyInside = isInFence(fenceId, userLon, userLat);
        String previousStatus = jedis.get(statusKey);

        if (previousStatus == null) {
            jedis.set(statusKey, currentlyInside ? "inside" : "outside");
            return;
        }

        boolean wasInside = "inside".equals(previousStatus);

        if (!wasInside && currentlyInside) {
            // 进入围栏
            onEnterFence(userId, fenceId);
            jedis.set(statusKey, "inside");
        } else if (wasInside && !currentlyInside) {
            // 离开围栏
            onLeaveFence(userId, fenceId);
            jedis.set(statusKey, "outside");
        }
    }

    private void onEnterFence(String userId, String fenceId) {
        System.out.println(userId + " 进入围栏 " + fenceId);
        // 推送通知、记录日志等
    }

    private void onLeaveFence(String userId, String fenceId) {
        System.out.println(userId + " 离开围栏 " + fenceId);
        // 推送通知、记录日志等
    }
}
```

### 4. 签到打卡

```java
public class CheckIn {
    private Jedis jedis;
    private static final String LOCATIONS_KEY = "checkin:locations";

    // 添加签到点
    public void addCheckInPoint(String pointId, double longitude, double latitude,
                                 String name, double radius) {
        jedis.geoadd(LOCATIONS_KEY, longitude, latitude, pointId);

        Map<String, String> info = new HashMap<>();
        info.put("name", name);
        info.put("radius", String.valueOf(radius));
        jedis.hmset("checkin:point:" + pointId, info);
    }

    // 用户签到
    public Map<String, Object> checkIn(String userId, double longitude, double latitude) {
        Map<String, Object> result = new HashMap<>();

        // 查找最近的签到点
        List<GeoRadiusResponse> points = jedis.georadius(
            LOCATIONS_KEY, longitude, latitude, 500, GeoUnit.M,
            GeoRadiusParam.geoRadiusParam()
                .withDist()
                .sortAscending()
                .count(1)
        );

        if (points.isEmpty()) {
            result.put("success", false);
            result.put("message", "附近没有签到点");
            return result;
        }

        GeoRadiusResponse nearest = points.get(0);
        String pointId = nearest.getMemberByString();
        double distance = nearest.getDistance();

        // 检查是否在签到范围内
        Map<String, String> pointInfo = jedis.hgetAll("checkin:point:" + pointId);
        double allowedRadius = Double.parseDouble(pointInfo.get("radius"));

        if (distance > allowedRadius) {
            result.put("success", false);
            result.put("message", "不在签到范围内，距离: " + distance + "米");
            return result;
        }

        // 记录签到
        String today = LocalDate.now().toString();
        String checkInKey = "checkin:record:" + today;
        jedis.hset(checkInKey, userId, pointId);
        jedis.expire(checkInKey, 86400 * 30);  // 保留 30 天

        result.put("success", true);
        result.put("pointId", pointId);
        result.put("pointName", pointInfo.get("name"));
        result.put("distance", distance);

        return result;
    }
}
```

## 性能优化

### 1. 使用 GeoHash 前缀筛选

GeoHash 相邻位置前缀相似，可用于粗筛：

```java
// 获取 GeoHash
String hash = jedis.geohash(KEY, "location1").get(0);

// 使用前 6 位进行粗筛
String prefix = hash.substring(0, 6);
```

### 2. 分区存储

按城市/区域分区，减少单个键的数据量：

```bash
# 按城市分区
GEOADD shops:beijing 116.403963 39.915119 "shop1"
GEOADD shops:shanghai 121.473701 31.230416 "shop2"

# 查询时只在对应分区搜索
GEORADIUS shops:beijing 116.403963 39.915119 5 km
```

### 3. 限制返回数量

```bash
# 限制返回 10 条
GEORADIUS key 116.403963 39.915119 5 km COUNT 10 ASC
```

### 4. 使用 Pipeline

批量操作时使用 Pipeline：

```java
Pipeline pipeline = jedis.pipelined();
for (Location loc : locations) {
    pipeline.geoadd(KEY, loc.getLon(), loc.getLat(), loc.getId());
}
pipeline.sync();
```

## 注意事项

### 1. 坐标范围

- 经度：-180 到 180
- 纬度：-85.05112878 到 85.05112878

超出范围会报错。

### 2. 精度

GeoHash 使用 52 位编码，精度约 0.6 米，满足大部分场景。

### 3. 删除操作

GEO 底层是 ZSET，使用 ZREM 删除：

```bash
ZREM restaurants "天安门"
```

### 4. 集群模式

在 Redis Cluster 中，所有位置必须在同一个槽位：

```bash
# 使用哈希标签
GEOADD {city}:shops:beijing 116.403963 39.915119 "shop1"
GEOADD {city}:shops:beijing 116.469392 39.95155 "shop2"
```

### 5. 距离计算

GEODIST 使用 Haversine 公式，假设地球是完美球体，实际误差约 0.5%。

## 与其他方案对比

| 方案               | 优势                 | 劣势         |
| ------------------ | -------------------- | ------------ |
| Redis GEO          | 简单、快速、低延迟   | 功能相对简单 |
| MongoDB            | 功能丰富，支持多边形 | 复杂度高     |
| PostgreSQL PostGIS | 专业 GIS 功能        | 部署复杂     |
| Elasticsearch      | 全文搜索+地理位置    | 资源消耗大   |

**推荐**：简单的附近搜索使用 Redis GEO，复杂地理分析使用专业 GIS 方案。

## 小结

Redis GEO 功能要点：

- ✅ 支持存储经纬度坐标
- ✅ 距离计算和范围查询
- ✅ 底层使用 Sorted Set（可用 ZSET 命令）
- ✅ 适合附近的人/店铺等场景
- ⚠️ 坐标范围有限制
- ⚠️ 复杂地理分析需要其他方案

GEO 功能为 LBS 应用提供了简单高效的解决方案！
