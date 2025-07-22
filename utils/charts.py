import matplotlib.pyplot as plt
import re

def generate_cbc_chart(cleaned_text):
    try:
        labels = []
        values = []
        pattern = r"(Haemoglobin|Hematocrit|Redcell count|MCV|MCH|MCHC|RDW|Platelet Count|T\.?L\.?C)\s+([\d.]+)"

        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        for match in matches:
            labels.append(match[0])
            values.append(float(match[1]))

        if not labels:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(labels, values, color='skyblue')
        ax.set_xlabel("Value")
        ax.set_title("CBC Report Overview")
        ax.invert_yaxis()
        plt.tight_layout()
        return fig
    except Exception as e:
        print("Chart generation error:", e)
        return None

def generate_charts_from_text(cleaned_text):
    return generate_cbc_chart(cleaned_text)
