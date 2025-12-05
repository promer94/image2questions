请帮我写一个 python 程序，用于将以下格式的 json 数据，生成为 word 中的多个表格

## 数据格式

```ts
interface Item {
  title: string
  options: {
    "a": string
    'b': string
    'c': string
    'd': string
  }
}

type Data = Array<Item>
```

## 表格格式要求

* 请使用 3 * 2 表格，合并第一行作为题目，剩下的表格作为选项
* 选项使用 A B C D 大写形式编号
* 表格样式为无边框


## 程序要求
* 使用 uv 来创建虚拟环境和依赖管理
* 使用 python-docx 来创建 word 文档和表格
