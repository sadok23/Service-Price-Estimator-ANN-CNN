import pandas as pd
import numpy as np
import random

categories = ['plumbing', 'electrical', 'gardening', 'cleaning']
difficulties = ['low', 'medium', 'high']

data = []
for _ in range(1000):
    category = random.choice(categories)
    difficulty = random.choice(difficulties)
    short_notice = random.randint(0, 1)
    weekend = random.randint(0, 1)
    distance_km = random.uniform(1, 30)  # 1–30 km
    review_score = round(random.uniform(3.0, 5.0), 2)
    past_jobs = random.randint(0, 200)

    base_price = {
        'plumbing': random.randint(70, 120),
        'electrical': random.randint(80, 150),
        'gardening': random.randint(40, 90),
        'cleaning': random.randint(30, 70)
    }[category]

    difficulty_multiplier = {
        'low': 1.0,
        'medium': 1.25,
        'high': 1.6
    }[difficulty]

    price = base_price * difficulty_multiplier

    price += distance_km * random.uniform(1.0, 3.0)  # TND per km

    if short_notice == 1:
        price += random.randint(15, 50)

    if weekend == 1:
        price *= random.uniform(1.20, 1.40)

    price += (review_score - 4.0) * random.uniform(2, 6)

    price += past_jobs * random.uniform(0.1, 0.3)

    price += random.uniform(-5, 5)

    data.append([
        category,
        difficulty,
        short_notice,
        weekend,
        round(distance_km, 2),
        review_score,
        past_jobs,
        round(price, 2)
    ])

df = pd.DataFrame(data, columns=[
    'category',
    'difficulty',
    'short_notice',
    'weekend',
    'distance_km',
    'review_score',
    'past_jobs',
    'price'
])

# Save to CSV
df.to_csv("service_price_dataset.csv", index=False)
print("Realistic dataset saved to 'service_price_dataset.csv'")
