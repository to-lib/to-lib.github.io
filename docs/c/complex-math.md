---
sidebar_position: 24
title: 复数与高级数学
---

# 复数与高级数学

C99 标准引入了对复数运算的原生支持，通过 `<complex.h>` 头文件提供。此外 `<math.h>` 也包含了大量高级数学函数。

## 复数运算 (<complex.h>)

### 基础类型

C 语言定义了三种复数类型：

- `double complex`: 对应 `double`
- `float complex`: 对应 `float`
- `long double complex`: 对应 `long double`

### 初始化与操作

使用宏 `I` 表示虚数单位 $i$。

```c
#include <stdio.h>
#include <complex.h>

int main(void) {
    // 定义复数 z = 3 + 4i
    double complex z1 = 3.0 + 4.0 * I;
    double complex z2 = 1.0 - 2.0 * I;

    // 基本运算
    double complex sum = z1 + z2;
    double complex diff = z1 - z2;
    double complex prod = z1 * z2;
    double complex quot = z1 / z2;

    // 打印复数 (注意格式化部分，creal取实部，cimag取虚部)
    printf("z1 = %.1f + %.1fi\n", creal(z1), cimag(z1));
    printf("Sum = %.1f + %.1fi\n", creal(sum), cimag(sum));

    return 0;
}
```

### 常用复数函数

| 函数         | 说明        | 示例                             |
| :----------- | :---------- | :------------------------------- |
| `creal(z)`   | 获取实部    | `double r = creal(z);`           |
| `cimag(z)`   | 获取虚部    | `double i = cimag(z);`           |
| `cabs(z)`    | 模 (绝对值) | `double mag = cabs(z);`          |
| `carg(z)`    | 辐角 (相位) | `double angle = carg(z);`        |
| `conj(z)`    | 共轭复数    | `double complex c = conj(z);`    |
| `cpow(z, w)` | 复数幂      | `double complex p = cpow(z, 2);` |
| `csqrt(z)`   | 复数开方    | `double complex s = csqrt(z);`   |
| `cexp(z)`    | 复数指数    | `double complex e = cexp(z);`    |

## 高级数学函数 (<math.h>)

除了基本的 `sin`, `cos`, `sqrt` 外，`<math.h>` 还有许多实用功能。

### 浮点环境与分类

```c
#include <math.h>
#include <stdio.h>

void check_float(double x) {
    if (isnan(x)) {
        printf("Not a Number (NaN)\n");
    } else if (isinf(x)) {
        printf("Infinity\n");
    } else if (isnormal(x)) {
        printf("Normal number\n");
    } else {
        printf("Subnormal or Zero\n");
    }
}

int main(void) {
    check_float(sqrt(-1.0)); // NaN
    check_float(1.0 / 0.0);  // Infinity
    return 0;
}
```

### 常用高级函数

- **`fmax(x, y)` / `fmin(x, y)`**: 浮点数最大/最小值。
- **`hypot(x, y)`**: 计算 $\sqrt{x^2 + y^2}$，防止中间结果溢出。
- **`cbrt(x)`**: 立方根。
- **`log2(x)`**: 以 2 为底的对数。
- **`fma(x, y, z)`**: 融合乘加 (Fused Multiply-Add)，计算 $x \times y + z$，只舍入一次，精度更高。

```c
double res = fma(x, y, z); // 比 x*y + z 更精确
```

- **`nextafter(x, y)`**: 返回 $x$ 之向 $y$ 方向的下一个可表示浮点数。这在数值分析中非常有用。

```c
printf("%.20f\n", nextafter(1.0, 2.0)); // 1.00000000000000022204...
```

## 总结

`<complex.h>` 和 `<math.h>` 为 C 语言提供了强大的科学计算能力。在进行信号处理、物理仿真或高精度计算时，熟练使用这些标准库函数可以事半功倍。
