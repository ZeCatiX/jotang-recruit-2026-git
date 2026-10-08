"""
从 Wikimedia Commons 下载猫狗图片数据集（使用缩略图绕过限流）。
优化版：使用多个搜索词、避免重复、支持断点续传。
"""
import hashlib
import json
import os
import time
import urllib.request
import urllib.parse
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CATS_DIR = DATA_DIR / "cats"
DOGS_DIR = DATA_DIR / "dogs"
TARGET = 200
HEADERS = {"User-Agent": "Mozilla/5.0 (task2-data-collection/1.0)"}
THUMB_SIZE = 384  # 缩略图大小（更小，更容易绕过限流）
SEEN_HASHES = set()


def ensure_dirs():
    for d in (CATS_DIR, DOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def api_call(url, timeout=15):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return None


def search_images(term, limit=50, offset=0):
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query"
        f"&list=search&srsearch={urllib.parse.quote(term)}"
        f"&srnamespace=6&srlimit={limit}&sroffset={offset}&format=json"
    )
    data = api_call(url)
    if data is None:
        return []
    return data.get("query", {}).get("search", [])


def get_thumb_url(title):
    """获取缩略图 URL"""
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query"
        f"&titles={urllib.parse.quote(title)}&prop=imageinfo"
        "&iiprop=url|mime&iiurlwidth=" + str(THUMB_SIZE) + "&format=json"
    )
    data = api_call(url, timeout=10)
    if data is None:
        return None, None
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    info = page.get("imageinfo", [{}])[0]
    thumb = info.get("thumburl", "")
    mime = info.get("mime", "")
    if thumb:
        return thumb.split("?")[0], mime
    return None, None


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def is_duplicate(path):
    h = file_hash(path)
    if h in SEEN_HASHES:
        return True
    SEEN_HASHES.add(h)
    return False


def download_image(url, save_path, max_size=500000):
    if save_path.exists():
        return False
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if len(data) > max_size or len(data) < 1000:
                return False
            save_path.write_bytes(data)
            # 验证不是重复文件
            if is_duplicate(save_path):
                save_path.unlink()
                return False
            return True
    except Exception:
        return False


def search_terms_for(category):
    if category == "cat":
        return [
            "kitten", "cat photo", "domestic cat", "cat face", "cat portrait",
            "orange cat", "black cat", "grey cat", "cat eyes", "cat close up",
            "cat sitting", "cat outdoors", "tiger cat", "tabby cat", "persian cat",
            "siamese cat", "ragdoll cat", "cat photo 2020", "cat 2019", "cat 2021",
        ]
    else:
        return [
            "puppy", "dog photo", "domestic dog", "dog face", "dog portrait",
            "golden retriever", "labrador", "corgi", "beagle", "poodle",
            "dog close up", "dog outdoors", "german shepherd", "husky",
            "shiba inu", "labrador retriever", "dog photo 2020", "dog 2019",
            "dog 2021", "chihuahua", "pug", "bull terrier",
        ]


def download_category(category, target=TARGET):
    out_dir = CATS_DIR if category == "cat" else DOGS_DIR

    # 统计已有文件
    existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
    downloaded = len(existing)
    print(f"  [{category}] 已有 {downloaded} 张")

    if downloaded >= target:
        return downloaded

    # 搜索策略：多关键词轮询
    terms = search_terms_for(category)
    term_idx = 0
    offset = 0
    max_total_offset = 2000
    total_offset = 0

    while downloaded < target and total_offset < max_total_offset:
        term = terms[term_idx % len(terms)]
        results = search_images(f"filetype:bitmap {term}", limit=50, offset=offset)

        for item in results:
            if downloaded >= target:
                break
            title = item.get("title", "")
            ext = title.lower().rsplit(".", 1)[-1]
            if ext not in ("jpg", "jpeg", "png"):
                continue

            thumb_url, mime = get_thumb_url(title)
            if not thumb_url:
                continue
            if mime and "image" not in mime:
                continue

            filename = f"{category}_{downloaded:04d}.{ext}"
            save_path = out_dir / filename

            if download_image(thumb_url, save_path):
                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"  [{category}] {downloaded}/{target}")

        offset += 50
        total_offset += 50
        if offset >= 500:
            offset = 0
            term_idx += 1
        time.sleep(0.2)

    return downloaded


def main():
    ensure_dirs()
    print("开始从 Wikimedia Commons 下载猫狗图片（缩略图模式）...")
    print(f"目标: 每类 {TARGET} 张, 缩略图 {THUMB_SIZE}px\n")

    cat_count = download_category("cat")
    dog_count = download_category("dog")

    cats = list(CATS_DIR.glob("*.jpg")) + list(CATS_DIR.glob("*.png"))
    dogs = list(DOGS_DIR.glob("*.jpg")) + list(DOGS_DIR.glob("*.png"))
    print(f"\n完成: cats={len(cats)}, dogs={len(dogs)}")


if __name__ == "__main__":
    main()
