# reddit_search_crawler_full.py
import os
import requests
import pandas as pd
from datetime import datetime, timezone
import time
import random

# ==============================
# ✅ SUBREDDITS 与 KEYWORDS 键对齐（新增了公司子版块）
# ==============================
SUBREDDITS = [
    "wallstreetbets", "options", "nvidia", "fednews",
    "microsoft", "apple", "amazon", "meta"
]

KEYWORDS = {
    "wallstreetbets": [
        "$qqq", "qqq", "qqq etf", "nasdaq 100", "ndx",
        "magnificent 7", "big tech", "tech stocks", "ai etf", "growth stocks",
        "$msft", "$nvda", "$aapl", "$amzn", "$meta",
        "semiconductor", "chips", "chip stocks",
        "ai compute", "computing power", "gpu",
        "data center", "ai infrastructure"
    ],
    "options": [
        "$qqq", "qqq", "ndx", "nq",
        "qqq option", "qqq weekly", "qqq straddle", "qqq iron condor",
        "ndx option", "nasdaq option",
        "iv rank", "implied volatility", "gamma exposure",
        "volatility crush", "theta decay",
        "$nvda option", "$msft option"
    ],
    "nvidia": [
        "$qqq", "qqq", "nasdaq", "nasdaq 100", "tech etf", "ai etf",
        "semiconductor", "chips", "chip demand", "chip shortage",
        "ai chips", "gpu", "blackwell", "hopper", "next-gen gpu",
        "ai compute", "compute capacity", "data center",
        "cloud ai", "training cluster",
        "qqq weight", "nasdaq exposure", "tech leadership"
    ],
    "fednews": [
        "qqq", "nasdaq", "tech stocks", "big tech", "growth stocks",
        "rate cut tech", "rate hike tech",
        "tech valuation", "pe multiple", "discount rate",
        "inflation tech", "real rates growth stocks",
        "rotation into tech", "tech rally", "tech selloff",
        "magnificent 7", "ai bubble", "ai valuation"
    ],
    "microsoft": [
        "ai", "artificial intelligence", "gpu", "data center", 
        "semiconductor", "chips", "tsmc", "interest rates", "fed meeting", 
        "fomc", "inflation", "earnings", "revenue", "guidance", "valuation", 
        "pe ratio", "tech rally", "selloff"
    ],
    "apple": [
        "ai", "artificial intelligence", "gpu", "data center", 
        "semiconductor", "chips", "tsmc", "interest rates", "fed meeting", 
        "fomc", "inflation", "earnings", "revenue", "guidance", "valuation", 
        "pe ratio", "tech rally", "selloff"
    ],
    "amazon": [
        "ai", "artificial intelligence", "gpu", "data center", 
        "semiconductor", "chips", "tsmc", "interest rates", "fed meeting", 
        "fomc", "inflation", "earnings", "revenue", "guidance", "valuation", 
        "pe ratio", "tech rally", "selloff"
    ],
    "meta": [
        "ai", "artificial intelligence", "gpu", "data center", 
        "semiconductor", "chips", "tsmc", "interest rates", "fed meeting", 
        "fomc", "inflation", "earnings", "revenue", "guidance", "valuation", 
        "pe ratio", "tech rally", "selloff"
    ],
}

# 时间范围：2023-01-01 至 2025-12-31 (UTC)
START_TS = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())
END_TS = int(datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())

OUTPUT_DIR = "data/reddit_search_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 安全的 User-Agent（模拟真实浏览器）
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
}

def search_reddit(subreddit, query, limit_per_keyword=150):
    """使用 Reddit 搜索接口抓取指定关键词的帖子（带指数退避和长延迟）"""
    all_posts = []
    after = None
    total_fetched = 0
    retry_count = 0
    max_retries = 5

    while total_fetched < limit_per_keyword and retry_count < max_retries:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            'q': query,
            'restrict_sr': 'on',   # 仅限当前 subreddit
            'sort': 'new',         # 按时间倒序（便于时间过滤）
            'limit': 100,
            't': 'all'             # 所有历史（由本地过滤时间）
        }
        if after:
            params['after'] = after

        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=15)
            
            if response.status_code == 429:
                wait_time = min(10 * (2 ** retry_count), 60)  # 指数退避，上限60秒
                print(f"⚠️ {subreddit}: 触发速率限制，等待 {wait_time} 秒...（重试 {retry_count + 1}/{max_retries}）")
                time.sleep(wait_time)
                retry_count += 1
                continue
            elif response.status_code != 200:
                print(f"❌ {subreddit} 请求失败: {response.status_code}")
                break

            # 成功响应，重置重试计数
            retry_count = 0

            data = response.json()
            posts = data['data']['children']
            if not posts:
                break

            for post in posts:
                p = post['data']
                created_utc = p['created_utc']
                
                # ⏱️ 严格时间过滤（只保留 2023–2025）
                if created_utc < START_TS or created_utc > END_TS:
                    continue
                
                all_posts.append({
                    'post_id': p['id'],
                    'subreddit': subreddit,
                    'title': p['title'],
                    'selftext': p.get('selftext', ''),
                    'score': p['score'],
                    'num_comments': p['num_comments'],
                    'created_utc': created_utc,
                    'url': f"https://reddit.com{p['permalink']}"
                })
            
            after = data['data'].get('after')
            if not after:
                break
                
            total_fetched += len(posts)
            # 🐢 关键：大幅延长请求间隔（Reddit 非常敏感）
            time.sleep(random.uniform(10, 15))  # 建议不低于 2.5 秒
            
        except Exception as e:
            print(f"❌ {subreddit} 抓取 '{query}' 时出错: {e}")
            time.sleep(5)
            retry_count += 1
    
    return all_posts

def main():
    print("🚀 开始使用完整关键词列表抓取 QQQ 相关 Reddit 数据（2023–2025）...")
    
    all_data = []
    
    for subreddit in SUBREDDITS:
        if subreddit not in KEYWORDS:
            print(f"⚠️ 跳过 subreddit '{subreddit}'：未定义关键词")
            continue
            
        keywords = KEYWORDS[subreddit]
        print(f"\n🔍 正在处理 r/{subreddit}（共 {len(keywords)} 个关键词）")
        
        seen_ids = set()
        subreddit_posts = []
        
        for kw in keywords:
            print(f"  - 搜索关键词: '{kw}'")
            posts = search_reddit(subreddit, kw, limit_per_keyword=150)
            
            # 去重（避免同一帖子被多个关键词命中多次）
            for post in posts:
                if post['post_id'] not in seen_ids:
                    subreddit_posts.append(post)
                    seen_ids.add(post['post_id'])
        
        print(f"  ✅ r/{subreddit} 共获得 {len(subreddit_posts)} 条唯一帖子")
        all_data.extend(subreddit_posts)
    
    # 全局去重（安全兜底）
    if not all_data:
        print("❌ 未抓取到任何数据！")
        return
        
    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=['post_id']).reset_index(drop=True)
    
    # 保存为 CSV（带 BOM，Excel 友好）
    output_file = os.path.join(OUTPUT_DIR, "qqq_reddit_2023_2025_full.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 总共抓取 {len(df)} 条唯一帖子")
    print(f"💾 已保存至: {output_file}")
    
    # 预览最近5条
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s', utc=True)
    print("\n📊 最近 5 条数据预览:")
    preview = df[['datetime', 'subreddit', 'title', 'score']].tail()
    print(preview.to_string(index=False, formatters={'datetime': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')}))

if __name__ == "__main__":
    main()