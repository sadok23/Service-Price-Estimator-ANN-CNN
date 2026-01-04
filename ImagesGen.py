import torch
from diffusers import StableDiffusionPipeline
import os, random

output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

categories = ["plumbing", "electrical", "cleaning", "gardening"]
difficulties = ["low", "medium", "high"]

prompts = {
    "plumbing": {
        "low": "Smartphone photo of a very small plumbing issue, tiny sink leak, few drops of water, minor damage, clean bathroom, simple task, clear lighting, realistic",
        "medium": "Photo of a noticeable plumbing problem, pipe leaking under sink, visible water puddle, moderate difficulty, tools needed, realistic handheld photo",
        "high": "Photo of a severe plumbing disaster, bathroom flooding, broken pipes shooting water, high difficulty, chaotic scene, water everywhere, emergency repair needed, realistic"
    },
    "electrical": {
        "low": "Smartphone photo of a small electrical issue, loose outlet cover, simple fix, no danger, clean wall, close-up, realistic",
        "medium": "Photo of a moderately damaged electrical outlet, burnt marks, exposed wires, medium risk, realistic handheld shot",
        "high": "Photo of a serious electrical hazard, heavily burnt electrical panel, melted wires, smoke damage, very dangerous, high difficulty repair, low lighting, realistic"
    },
    "cleaning": {
        "low": "Photo of a slightly messy room, few items on the floor, light cleaning needed, easy task, natural light, realistic smartphone photo",
        "medium": "Photo of a very messy kitchen, dirty dishes, spills, clutter, medium cleaning workload, handheld smartphone photo",
        "high": "Photo of an extremely dirty bathroom, mold everywhere, heavy stains, trash on the floor, overwhelming mess, high cleaning difficulty, realistic"
    },
    "gardening": {
        "low": "Photo of a small backyard with a few weeds, slightly overgrown plants, easy gardening task, sunny day, realistic lighting",
        "medium": "Photo of an overgrown lawn, long grass, scattered leaves, moderate difficulty yard work, handheld smartphone photo",
        "high": "Photo of a severely overgrown garden jungle, tall grass, dead plants, branches everywhere, very difficult gardening task, chaotic, realistic lighting"
    }
}

num_images_per_class = 200  

view_angles = ["close-up", "wide-angle", "from above", "from below", "tilted angle"]
lighting = ["soft light", "harsh shadows", "natural daylight", "low light", "warm lighting"]
camera_styles = ["smartphone photo", "handheld photo", "mobile shot", "slightly blurry", "realistic photo"]

for diff in difficulties:
    for cat in categories:
        os.makedirs(os.path.join(output_dir, diff, cat), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

for diff in difficulties:
    for cat in categories:
        print(f"🔹 Generating images for {cat} - {diff}...")
        base_prompt = prompts[cat][diff]
        save_path = os.path.join(output_dir, diff, cat)

        for i in range(num_images_per_class):
            prompt = (
                base_prompt + ", " +
                random.choice(view_angles) + ", " +
                random.choice(lighting) + ", " +
                random.choice(camera_styles)
            )

            generator = torch.Generator(device=device).manual_seed(random.randint(1, 2**31-1))
            image = pipe(prompt, guidance_scale=random.uniform(6.0, 9.0), generator=generator).images[0]
            image.save(os.path.join(save_path, f"{cat}_{diff}_{i+1:04d}.png"))

print("All images generated successfully!")
