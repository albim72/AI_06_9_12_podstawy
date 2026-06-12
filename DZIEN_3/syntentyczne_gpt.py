from faker import Faker
import random
import csv


fake = Faker("pl_PL")


# 16 dodatkowych kolorów z palety RGB
OTHER_COLORS = [
    ("niebieski", "#0000FF"),
    ("żółty", "#FFFF00"),
    ("czarny", "#000000"),
    ("szary", "#808080"),
    ("pomarańczowy", "#FFA500"),
    ("fioletowy", "#800080"),
    ("różowy", "#FFC0CB"),
    ("brązowy", "#A52A2A"),
    ("turkusowy", "#40E0D0"),
    ("granatowy", "#000080"),
    ("limonkowy", "#00FF00"),
    ("oliwkowy", "#808000"),
    ("srebrny", "#C0C0C0"),
    ("beżowy", "#F5F5DC"),
    ("bordowy", "#800000"),
    ("lawendowy", "#E6E6FA"),
]


BASE_COLORS = [
    ("czerwony", "#FF0000", 10),
    ("zielony", "#008000", 40),
    ("złoty", "#FFD700", 10),
    ("biały", "#FFFFFF", 20),
]


def generate_age() -> int:
    """
    Generuje wiek zgodnie z rozkładem:
    - 18–30: 17%
    - 31–60: 43%
    - 61–99: 40%
    """
    group = random.choices(
        population=["18_30", "31_60", "61_99"],
        weights=[17, 43, 40],
        k=1
    )[0]

    if group == "18_30":
        return random.randint(18, 30)
    elif group == "31_60":
        return random.randint(31, 60)
    else:
        return random.randint(61, 99)


def generate_amount() -> float:
    """
    Generuje kwotę z rozkładu normalnego:
    średnia = 560 zł
    odchylenie standardowe = 101 zł
    """
    amount = random.gauss(mu=560, sigma=101)

    # Zabezpieczenie przed ujemnymi wartościami
    amount = max(0, amount)

    return round(amount, 2)


def generate_color() -> tuple[str, str]:
    """
    Generuje kolor zgodnie z wagami:
    - czerwony: 10%
    - zielony: 40%
    - złoty: 10%
    - biały: 20%
    - pozostałe 16 kolorów razem: 20%
    """
    colors = BASE_COLORS.copy()

    # Każdy z 16 dodatkowych kolorów dostaje równą część z pozostałych 20%
    other_color_weight = 20 / len(OTHER_COLORS)

    for name, rgb in OTHER_COLORS:
        colors.append((name, rgb, other_color_weight))

    selected = random.choices(
        population=colors,
        weights=[color[2] for color in colors],
        k=1
    )[0]

    return selected[0], selected[1]


def generate_record() -> dict:
    """
    Generuje jeden rekord danych syntetycznych.
    """
    color_name, color_rgb = generate_color()

    return {
        "imie": fake.first_name(),
        "nazwisko": fake.last_name(),
        "miasto": fake.city(),
        "wiek": generate_age(),
        "kwota": generate_amount(),
        "kolor": color_name,
        "kolor_rgb": color_rgb,
    }


def generate_dataset(number_of_records: int, output_file: str) -> None:
    """
    Generuje zbiór danych i zapisuje go do pliku CSV.
    """
    records = [generate_record() for _ in range(number_of_records)]

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "imie",
                "nazwisko",
                "miasto",
                "wiek",
                "kwota",
                "kolor",
                "kolor_rgb",
            ]
        )

        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    generate_dataset(
        number_of_records=1000,
        output_file="dane_syntetyczne.csv"
    )

    print("Wygenerowano plik: dane_syntetyczne.csv")
