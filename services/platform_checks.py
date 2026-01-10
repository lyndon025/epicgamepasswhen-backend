def check_pc_platform(platforms_data, platform_name):
    if not platforms_data or len(platforms_data) == 0:
        return None

    try:
        pc_keywords = ["pc", "windows", "linux", "macos"]
        platform_names = [
            p.get("platform", {}).get("name", "").lower()
            for p in platforms_data
            if p and isinstance(p, dict)
        ]

        if not platform_names:
            return None

        is_pc = any(keyword in " ".join(platform_names) for keyword in pc_keywords)

        if not is_pc:
            return {
                "tier": "Platform Check",
                "category": "not on pc (console exclusive)",
                "confidence": 95,
                "reasoning": f"Game is not available on PC. Available on: {', '.join([p for p in platform_names if p])}. Epic Games Store only offers PC games.",
                "platforms": platform_names,
            }
    except Exception as e:
        print(f"Platform check error (Epic): {e}")
        return None

    return None


def check_xbox_platform(platforms_data, platform_name):
    if not platforms_data or len(platforms_data) == 0:
        return None

    try:
        xbox_keywords = ["xbox", "pc", "windows"]
        platform_names = [
            p.get("platform", {}).get("name", "").lower()
            for p in platforms_data
            if p and isinstance(p, dict)
        ]

        if not platform_names:
            return None

        is_xbox = any(keyword in " ".join(platform_names) for keyword in xbox_keywords)

        if not is_xbox:
            return {
                "tier": "Platform Check",
                "category": "not on xbox/pc",
                "confidence": 95,
                "reasoning": f"Game is not available on Xbox or PC. Available on: {', '.join([p for p in platform_names if p])}. Xbox Game Pass requires Xbox or PC platform.",
                "platforms": platform_names,
            }
    except Exception as e:
        print(f"Platform check error (Xbox): {e}")
        return None

    return None


def check_playstation_platform(platforms_data, platform_name):
    if not platforms_data or len(platforms_data) == 0:
        return None

    try:
        ps_keywords = ["playstation", "ps4", "ps5"]
        platform_names = [
            p.get("platform", {}).get("name", "").lower()
            for p in platforms_data
            if p and isinstance(p, dict)
        ]

        if not platform_names:
            return None

        is_ps = any(keyword in " ".join(platform_names) for keyword in ps_keywords)

        if not is_ps:
            return {
                "tier": "Platform Check",
                "category": "not on playstation",
                "confidence": 95,
                "reasoning": f"Game is not available on PlayStation. Available on: {', '.join([p for p in platform_names if p])}. PS Plus requires PlayStation platform.",
                "platforms": platform_names,
            }
    except Exception as e:
        print(f"Platform check error (PS): {e}")
        return None

    return None
