from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from pypinyin import Style, lazy_pinyin


KEYWORD = "安居小宝"
CATEGORY_COUNTS = {
    "similar_pronunciation": 400,
    "smart_home_command": 250,
    "daily_sentence": 200,
    "other_wakeword": 100,
    "filler_short": 50,
}
SPEEDS = [0.9, 1.0, 1.08]


def normalize_text(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?；;：:（）()《》<>\"'“”‘’\-_/\\]+", "", text)


def pinyin_signature(text: str) -> tuple[str, ...]:
    return tuple(lazy_pinyin(normalize_text(text), style=Style.NORMAL, errors="ignore"))


KEYWORD_PINYIN = pinyin_signature(KEYWORD)
BLOCKED_SURFACE = {
    KEYWORD,
    "安居晓宝",
    "安居小饱",
    "安居小保",
    "安居小包",
    "安居小堡",
    "安居小豹",
    "安居小报",
    "安居小鲍",
    "安居小褓",
}


def contains_keyword_pinyin(sig: tuple[str, ...]) -> bool:
    if len(sig) < len(KEYWORD_PINYIN):
        return False
    for i in range(0, len(sig) - len(KEYWORD_PINYIN) + 1):
        if sig[i:i + len(KEYWORD_PINYIN)] == KEYWORD_PINYIN:
            return True
    return False


def reject_reason(text: str) -> str | None:
    norm = normalize_text(text)
    if not norm:
        return "empty_text"
    if norm in BLOCKED_SURFACE:
        return "blocked_surface"
    if KEYWORD in norm:
        return "contains_keyword_text"
    sig = pinyin_signature(norm)
    if sig == KEYWORD_PINYIN:
        return "same_pinyin_as_keyword"
    if contains_keyword_pinyin(sig):
        return "contains_keyword_pinyin"
    return None


def expand_similar_candidates() -> list[str]:
    first_parts = [
        "按住", "安住", "安珠", "安主", "安助", "安竹", "安逐", "安注",
        "安静", "安吉", "安集", "安急", "安记", "安区", "安去", "安久",
        "安聚", "安具", "安菊", "安局", "安居家", "安居的", "安居一",
        "安居新", "安居好", "安居老", "安居大", "安居门", "安居灯",
    ]
    second_parts = [
        "小贝", "小北", "小白", "小拜", "小班", "小板", "小帮", "小棒",
        "小本", "小笨", "小兵", "小冰", "小标", "小表", "小播", "小伯",
        "小波", "小布", "小不", "小度", "小杜", "小米", "小明", "小敏",
        "小猫", "小麦", "小门", "小梦", "小智", "小志", "小周", "小钟",
        "小陈", "小成", "小王", "小张", "小李", "小刘", "小赵", "小孙",
    ]
    candidates = []
    for a in first_parts:
        for b in second_parts:
            candidates.append(a + b)
    candidates.extend([
        "安居宝贝", "安居宝宝", "安居小助手", "安居管家", "安居小屏",
        "安居小窗", "安居小灯", "安居小站", "安居小厅", "安居小屋",
        "按住小宝键", "按住小按钮", "安居宝打开", "安居帮我看一下",
    ])
    return candidates


def expand_smart_home_candidates() -> list[str]:
    rooms = ["客厅", "卧室", "书房", "厨房", "阳台", "主卧", "次卧", "玄关", "餐厅", "卫生间"]
    devices = ["灯", "空调", "窗帘", "电视", "风扇", "加湿器", "净化器", "地暖", "热水器", "插座"]
    actions = [
        "打开", "关闭", "调亮", "调暗", "调高一点", "调低一点", "切换模式", "暂停",
        "继续运行", "设为自动", "设为睡眠模式", "定时十分钟", "检查状态",
    ]
    candidates = []
    for room in rooms:
        for device in devices:
            for action in actions:
                candidates.append(f"{action}{room}{device}")
                candidates.append(f"把{room}{device}{action}")
    candidates.extend([
        "现在温度是多少", "空气质量怎么样", "切换到制冷", "切换到制热",
        "帮我打开回家模式", "打开离家模式", "关闭所有灯光", "播放一首歌",
        "音量调大一点", "音量调小一点", "明早七点提醒我", "取消所有定时",
    ])
    return candidates


def expand_daily_candidates() -> list[str]:
    starts = ["今天", "明天", "刚才", "晚上", "早上", "中午", "周末", "等会儿", "现在", "一会儿"]
    verbs = ["去公司", "回家", "开会", "买菜", "取快递", "看电影", "打电话", "发消息", "整理资料", "准备出门"]
    tails = ["可以吗", "有点晚", "不用着急", "记得提醒我", "我们再说", "先这样", "路上注意", "早点休息"]
    candidates = []
    for s in starts:
        for v in verbs:
            for t in tails:
                candidates.append(f"{s}{v}{t}")
    candidates.extend([
        "你听见了吗", "我刚刚说到哪里了", "这个问题稍后处理", "把文件放在桌上",
        "我们下午再确认", "明天早一点出门", "这个方案还要再看", "先把门关上",
        "这件事不用担心", "我马上就回来",
    ])
    return candidates


def expand_other_wakeword_candidates() -> list[str]:
    names = [
        "小爱同学", "小度小度", "天猫精灵", "小艺小艺", "小布小布", "你好小迪",
        "小白小白", "小智小智", "小米小米", "你好小乐", "你好小微", "你好小云",
        "小助手你好", "智能管家", "语音助手", "小管家小管家", "小鱼小鱼",
        "小蓝小蓝", "小方小方", "你好小美", "你好小林", "你好小安",
        "你好小新", "你好小佳", "小精灵小精灵", "小音箱小音箱",
        "家居助手", "智能音箱", "家庭助手", "语音管家",
    ]
    actions = [
        "打开客厅灯", "今天天气怎么样", "播放音乐", "关闭空调", "调高音量",
        "设置闹钟", "打开窗帘", "讲个故事",
    ]
    candidates = []
    for name in names:
        candidates.append(name)
        for action in actions:
            candidates.append(f"{name}{action}")
    return candidates


def expand_filler_candidates() -> list[str]:
    base = [
        "嗯", "啊", "哦", "喂", "好", "行", "可以", "不用", "等一下", "稍等",
        "这个", "那个", "然后", "就是", "你说", "我在", "听到了", "知道了",
        "没事", "好的", "对", "不对", "再来", "继续", "停一下",
    ]
    candidates = []
    for word in base:
        candidates.append(word)
        candidates.append(word + "啊")
    return candidates


def unique_filtered(candidates: list[str]) -> tuple[list[str], list[dict]]:
    accepted = []
    rejected = []
    seen = set()
    for text in candidates:
        text = normalize_text(text)
        if text in seen:
            continue
        seen.add(text)
        reason = reject_reason(text)
        if reason is None:
            accepted.append(text)
        else:
            rejected.append({
                "text": text,
                "reason": reason,
                "pinyin": " ".join(pinyin_signature(text)),
            })
    return accepted, rejected


def read_prompts(path: Path) -> list[dict]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    if not prompts:
        raise ValueError(f"no prompt rows in {path}")
    return prompts


def build_plan(prompt_manifest: Path, out_dir: Path, seed: int) -> dict:
    rng = random.Random(seed)
    prompts = read_prompts(prompt_manifest)
    pools = {
        "similar_pronunciation": expand_similar_candidates(),
        "smart_home_command": expand_smart_home_candidates(),
        "daily_sentence": expand_daily_candidates(),
        "other_wakeword": expand_other_wakeword_candidates(),
        "filler_short": expand_filler_candidates(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rejected = []
    selected = []

    for category, need in CATEGORY_COUNTS.items():
        accepted, rejected = unique_filtered(pools[category])
        all_rejected.extend({"category": category, **row} for row in rejected)
        if len(accepted) < need:
            raise ValueError(f"{category} has only {len(accepted)} safe candidates, need {need}")
        rng.shuffle(accepted)
        for text in accepted[:need]:
            selected.append({
                "text": text,
                "label_type": "negative",
                "category": category,
            })

    rng.shuffle(selected)
    plan_path = out_dir / "text_plan.jsonl"
    with plan_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(selected, 1):
            prompt = prompts[(idx - 1) % len(prompts)]
            speed = SPEEDS[(idx - 1) % len(SPEEDS)]
            record = {
                "index": idx,
                **row,
                "prompt_id": prompt["id"],
                "prompt_text": prompt["text"],
                "prompt_wav": prompt["wav"],
                "prompt_speaker": prompt.get("speaker", ""),
                "prompt_gender": prompt.get("gender", ""),
                "prompt_age_group": prompt.get("age_group", ""),
                "prompt_accent": prompt.get("accent", ""),
                "speed": speed,
                "pinyin": " ".join(pinyin_signature(row["text"])),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    rejected_path = out_dir / "rejected_text_candidates.jsonl"
    with rejected_path.open("w", encoding="utf-8") as f:
        for row in all_rejected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "keyword": KEYWORD,
        "keyword_pinyin": " ".join(KEYWORD_PINYIN),
        "count": len(selected),
        "category_counts": CATEGORY_COUNTS,
        "rejected_candidate_count": len(all_rejected),
        "text_plan": str(plan_path),
        "rejected_text_candidates": str(rejected_path),
    }
    (out_dir / "text_plan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_manifest", default=r"E:\CodeWorking\Dataset\anju_xiaobao_cosyvoice2_500_gpu\prompt_voices.jsonl")
    parser.add_argument("--output_dir", default=r"E:\CodeWorking\Dataset\anju_xiaobao_negative_cosyvoice2_1000_gpu")
    parser.add_argument("--seed", type=int, default=20260507)
    args = parser.parse_args()
    summary = build_plan(Path(args.prompt_manifest), Path(args.output_dir), args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
