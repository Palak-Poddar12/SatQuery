from models.base_inference import infer as base_infer


PROMPTS = {
    "brief": "Describe this satellite image in one sentence.",
    "detailed": "Describe land cover, objects, and features in detail.",
    "technical": "Describe spectral characteristics and land cover classes.",
}


def run_caption(image_b64: str, style: str = "detailed") -> dict:
    question = PROMPTS.get(style, PROMPTS["detailed"])
    result = base_infer(image_b64, question)

    # Extract land-cover labels from the generated answer.
    land_cover = [
        word
        for word in result["answer"].split()
        if word.lower()
        in [
            "forest",
            "urban",
            "water",
            "farmland",
            "residential",
            "industrial",
            "wetland",
            "soil",
        ]
    ]

    result["caption"] = result.pop("answer")
    result["land_cover"] = land_cover
    result["style"] = style
    result["task_type"] = "caption"
    return result
