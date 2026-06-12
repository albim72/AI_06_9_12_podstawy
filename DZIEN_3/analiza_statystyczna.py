import numpy as np
import pandas as pd


INPUT_FILE = "dane_syntetyczne.csv"
OUTPUT_FILE = "analiza_danych_syntetycznych.xlsx"


def assign_age_group(age: int) -> str:
    """
    Przypisuje osobę do grupy wiekowej.
    """
    if 18 <= age <= 30:
        return "18-30"
    elif 31 <= age <= 60:
        return "31-60"
    else:
        return "61+"


def create_histogram(series: pd.Series, bins: int = 10) -> pd.DataFrame:
    """
    Tworzy histogram dla danych liczbowych.
    """
    counts, bin_edges = np.histogram(series.dropna(), bins=bins)

    return pd.DataFrame({
        "przedzial_od": bin_edges[:-1].round(2),
        "przedzial_do": bin_edges[1:].round(2),
        "liczba_obserwacji": counts
    })


def analyze_data(input_file: str, output_file: str) -> None:
    """
    Wczytuje dane, przeprowadza analizę statystyczną
    i zapisuje wyniki do pliku Excel.
    """

    df = pd.read_csv(input_file)

    # Uporządkowanie nazw kolumn, gdyby były przypadkowe spacje
    df.columns = df.columns.str.strip()

    # Dodanie grupy wiekowej
    df["grupa_wiekowa"] = df["wiek"].apply(assign_age_group)

    # -----------------------------
    # 1. Podstawowe informacje
    # -----------------------------

    summary = pd.DataFrame({
        "metryka": [
            "liczba_rekordow",
            "liczba_kolumn",
            "liczba_brakow_danych",
            "liczba_duplikatow",
            "sredni_wiek",
            "mediana_wieku",
            "minimalny_wiek",
            "maksymalny_wiek",
            "srednia_kwota",
            "mediana_kwoty",
            "minimalna_kwota",
            "maksymalna_kwota",
            "odchylenie_standardowe_kwoty",
        ],
        "wartosc": [
            len(df),
            len(df.columns),
            df.isna().sum().sum(),
            df.duplicated().sum(),
            df["wiek"].mean(),
            df["wiek"].median(),
            df["wiek"].min(),
            df["wiek"].max(),
            df["kwota"].mean(),
            df["kwota"].median(),
            df["kwota"].min(),
            df["kwota"].max(),
            df["kwota"].std(),
        ]
    })

    # -----------------------------
    # 2. Statystyki opisowe
    # -----------------------------

    numeric_stats = df[["wiek", "kwota"]].describe().T
    numeric_stats["wariancja"] = df[["wiek", "kwota"]].var()
    numeric_stats["skosnosc"] = df[["wiek", "kwota"]].skew()
    numeric_stats["kurtoza"] = df[["wiek", "kwota"]].kurtosis()

    # -----------------------------
    # 3. Analiza wieku
    # -----------------------------

    age_group_counts = (
        df["grupa_wiekowa"]
        .value_counts()
        .rename_axis("grupa_wiekowa")
        .reset_index(name="liczba")
    )

    age_group_counts["procent"] = (
        age_group_counts["liczba"] / len(df) * 100
    ).round(2)

    expected_age_distribution = pd.DataFrame({
        "grupa_wiekowa": ["18-30", "31-60", "61+"],
        "oczekiwany_procent": [17, 43, 40]
    })

    age_group_comparison = age_group_counts.merge(
        expected_age_distribution,
        on="grupa_wiekowa",
        how="left"
    )

    age_group_comparison["roznica_pp"] = (
        age_group_comparison["procent"]
        - age_group_comparison["oczekiwany_procent"]
    ).round(2)

    age_histogram = create_histogram(df["wiek"], bins=10)

    # -----------------------------
    # 4. Analiza kwoty
    # -----------------------------

    amount_histogram = create_histogram(df["kwota"], bins=12)

    amount_percentiles = pd.DataFrame({
        "percentyl": ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"],
        "wartosc": np.percentile(
            df["kwota"].dropna(),
            [1, 5, 10, 25, 50, 75, 90, 95, 99]
        ).round(2)
    })

    # Odstające wartości kwoty metodą IQR
    q1 = df["kwota"].quantile(0.25)
    q3 = df["kwota"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers_iqr = df[
        (df["kwota"] < lower_bound) |
        (df["kwota"] > upper_bound)
    ].copy()

    outliers_iqr["typ_odstajacej"] = np.where(
        outliers_iqr["kwota"] < lower_bound,
        "ponizej_dolnej_granicy",
        "powyzej_gornej_granicy"
    )

    # Odstające wartości kwoty metodą z-score
    df["kwota_z_score"] = (
        df["kwota"] - df["kwota"].mean()
    ) / df["kwota"].std()

    outliers_z_score = df[
        df["kwota_z_score"].abs() > 3
    ].copy()

    # -----------------------------
    # 5. Analiza kolorów
    # -----------------------------

    color_counts = (
        df["kolor"]
        .value_counts()
        .rename_axis("kolor")
        .reset_index(name="liczba")
    )

    color_counts["procent"] = (
        color_counts["liczba"] / len(df) * 100
    ).round(2)

    expected_color_distribution = pd.DataFrame({
        "kolor": ["czerwony", "zielony", "złoty", "biały", "inne_lacznie"],
        "oczekiwany_procent": [10, 40, 10, 20, 20]
    })

    main_colors = ["czerwony", "zielony", "złoty", "biały"]

    other_colors_summary = pd.DataFrame({
        "kolor": ["inne_lacznie"],
        "liczba": [df[~df["kolor"].isin(main_colors)].shape[0]]
    })

    other_colors_summary["procent"] = (
        other_colors_summary["liczba"] / len(df) * 100
    ).round(2)

    main_color_summary = color_counts[
        color_counts["kolor"].isin(main_colors)
    ]

    color_comparison = pd.concat(
        [main_color_summary, other_colors_summary],
        ignore_index=True
    )

    color_comparison = color_comparison.merge(
        expected_color_distribution,
        on="kolor",
        how="left"
    )

    color_comparison["roznica_pp"] = (
        color_comparison["procent"]
        - color_comparison["oczekiwany_procent"]
    ).round(2)

    # -----------------------------
    # 6. Analiza miast
    # -----------------------------

    city_counts = (
        df["miasto"]
        .value_counts()
        .head(20)
        .rename_axis("miasto")
        .reset_index(name="liczba")
    )

    city_counts["procent"] = (
        city_counts["liczba"] / len(df) * 100
    ).round(2)

    # -----------------------------
    # 7. Analiza krzyżowa
    # -----------------------------

    age_color_cross = pd.crosstab(
        df["grupa_wiekowa"],
        df["kolor"],
        margins=True
    )

    age_group_amount = (
        df
        .groupby("grupa_wiekowa")
        .agg(
            liczba=("kwota", "count"),
            srednia_kwota=("kwota", "mean"),
            mediana_kwoty=("kwota", "median"),
            min_kwota=("kwota", "min"),
            max_kwota=("kwota", "max"),
            odchylenie_std_kwoty=("kwota", "std"),
            sredni_wiek=("wiek", "mean")
        )
        .round(2)
        .reset_index()
    )

    color_amount = (
        df
        .groupby("kolor")
        .agg(
            liczba=("kwota", "count"),
            srednia_kwota=("kwota", "mean"),
            mediana_kwoty=("kwota", "median"),
            min_kwota=("kwota", "min"),
            max_kwota=("kwota", "max"),
            odchylenie_std_kwoty=("kwota", "std")
        )
        .round(2)
        .reset_index()
        .sort_values("liczba", ascending=False)
    )

    # -----------------------------
    # 8. Korelacje
    # -----------------------------

    correlation_matrix = df[["wiek", "kwota"]].corr().round(4)

    # -----------------------------
    # 9. Braki danych
    # -----------------------------

    missing_data = pd.DataFrame({
        "kolumna": df.columns,
        "liczba_brakow": df.isna().sum().values,
        "procent_brakow": (df.isna().sum().values / len(df) * 100).round(2)
    })

    # -----------------------------
    # 10. Zapis do Excela
    # -----------------------------

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Dane", index=False)

        summary.to_excel(writer, sheet_name="Podsumowanie", index=False)
        numeric_stats.to_excel(writer, sheet_name="Statystyki_numeryczne")
        missing_data.to_excel(writer, sheet_name="Braki_danych", index=False)

        age_group_comparison.to_excel(
            writer,
            sheet_name="Wiek_grupy",
            index=False
        )

        age_histogram.to_excel(
            writer,
            sheet_name="Wiek_histogram",
            index=False
        )

        amount_histogram.to_excel(
            writer,
            sheet_name="Kwota_histogram",
            index=False
        )

        amount_percentiles.to_excel(
            writer,
            sheet_name="Kwota_percentyle",
            index=False
        )

        color_counts.to_excel(
            writer,
            sheet_name="Kolory",
            index=False
        )

        color_comparison.to_excel(
            writer,
            sheet_name="Kolory_porownanie",
            index=False
        )

        city_counts.to_excel(
            writer,
            sheet_name="Miasta_TOP20",
            index=False
        )

        age_color_cross.to_excel(
            writer,
            sheet_name="Wiek_kolor_tabela"
        )

        age_group_amount.to_excel(
            writer,
            sheet_name="Wiek_kwota",
            index=False
        )

        color_amount.to_excel(
            writer,
            sheet_name="Kolor_kwota",
            index=False
        )

        correlation_matrix.to_excel(
            writer,
            sheet_name="Korelacje"
        )

        outliers_iqr.to_excel(
            writer,
            sheet_name="Odstajace_IQR",
            index=False
        )

        outliers_z_score.to_excel(
            writer,
            sheet_name="Odstajace_zscore",
            index=False
        )

    print(f"Analiza zakończona. Wyniki zapisano do pliku: {output_file}")


if __name__ == "__main__":
    analyze_data(INPUT_FILE, OUTPUT_FILE)
