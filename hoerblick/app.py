from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from hoerblick.dataset import dataset_status
from hoerblick.evaluation import load_comparison_table
from hoerblick.models import ModelResult, get_model_adapter, model_status
from hoerblick.priority import WARNING_LEVELS, warning_for_prediction


PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="HörBlick",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --hb-ink: #1d2433;
            --hb-muted: #5d6677;
            --hb-surface: #f6f7fb;
            --hb-panel: #ffffff;
            --hb-primary: #3559d5;
            --hb-primary-soft: #e5eaff;
            --hb-coral: #e15b4e;
            --hb-amber: #c78312;
            --hb-mint: #168474;
            --hb-line: #d9deea;
        }

        .stApp {
            background: var(--hb-surface);
            color: var(--hb-ink);
        }

        [data-testid="stSidebar"] {
            background: #202b43;
        }

        [data-testid="stSidebar"] * {
            color: #f4f6ff;
        }

        [data-testid="stSidebar"] .stRadio > label {
            color: #cfd7ed;
            font-weight: 700;
        }

        .hb-kicker {
            color: var(--hb-primary);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hb-title {
            color: var(--hb-ink);
            font-size: 3.4rem;
            font-weight: 850;
            line-height: 0.98;
            margin: 0;
            letter-spacing: 0;
        }

        @media (max-width: 700px) {
            .hb-title {
                font-size: 2.4rem;
            }
        }

        .hb-subtitle {
            color: var(--hb-muted);
            font-size: 1.08rem;
            line-height: 1.55;
            max-width: 760px;
            margin-top: 1rem;
        }

        .hb-panel {
            background: var(--hb-panel);
            border: 1px solid var(--hb-line);
            border-radius: 10px;
            padding: 1.25rem;
            box-shadow: 0 12px 28px rgba(31, 42, 67, 0.07);
        }

        .hb-panel h3 {
            margin-top: 0;
            color: var(--hb-ink);
        }

        .hb-mini-label {
            color: var(--hb-muted);
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .hb-status {
            border-left: 5px solid var(--hb-primary);
            background: var(--hb-primary-soft);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--hb-ink);
        }

        .hb-warning {
            border-radius: 10px;
            padding: 1.35rem 1.4rem;
            color: #ffffff;
            min-height: 154px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .hb-warning .level {
            font-size: 0.8rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.9;
        }

        .hb-warning .headline {
            font-size: 1.7rem;
            font-weight: 850;
            line-height: 1.05;
            margin-top: 0.35rem;
        }

        .hb-warning .action {
            font-size: 1rem;
            margin-top: 0.45rem;
        }

        .hb-warning.rot { background: var(--hb-coral); }
        .hb-warning.gelb { background: var(--hb-amber); }
        .hb-warning.gruen { background: var(--hb-mint); }

        .hb-footer {
            color: var(--hb-muted);
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--hb-line);
        }

        div[data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #9aa8d1;
            background: #f8f9ff;
            border-radius: 10px;
        }

        .stButton > button {
            border-radius: 999px;
            min-height: 2.75rem;
            font-weight: 800;
            border: 1px solid #b7c3ef;
        }

        .stDownloadButton > button {
            border-radius: 999px;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown('<div class="hb-kicker">Klausurersatzleistung Informatik</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hb-title">HörBlick</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hb-subtitle">Umgebungsgeräusche erkennen, einordnen und für Menschen mit Hörbeeinträchtigung sichtbar machen.</p>',
        unsafe_allow_html=True,
    )


def render_warning(result: ModelResult) -> None:
    warning = warning_for_prediction(result.label, result.confidence)
    css_class = warning["level"].lower()
    st.markdown(
        f"""
        <div class="hb-warning {css_class}">
            <div class="level">{warning["icon"]} Warnstufe {warning["level"]}</div>
            <div class="headline">{warning["title"]}</div>
            <div class="action">{warning["action"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    left.metric("Erkannte Klasse", result.label)
    right.metric("Konfidenz", f"{result.confidence:.0%}")
    st.progress(float(result.confidence), text=f"Modellkonfidenz: {result.confidence:.0%}")
    if result.is_demo:
        st.info(
            "Demo-Modus: Für die echte Bewertung müssen die exportierte Modell-Datei "
            "und die optionalen ML-Abhängigkeiten ergänzt werden."
        )


def analyse_page() -> None:
    render_header()
    st.markdown(
        '<div class="hb-status"><strong>Prüfungsziel:</strong> Eine Audiodatei wird mit Modell A oder Modell B analysiert und als klare Warnstufe ausgegeben.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    upload_col, model_col = st.columns([1.25, 0.75], gap="large")
    with upload_col:
        st.subheader("Audiodatei auswählen")
        uploaded_file = st.file_uploader(
            "WAV, MP3 oder FLAC",
            type=["wav", "mp3", "flac"],
            help="Für die Prüfung werden dieselben Dateien für beide Modellvarianten verwendet.",
        )

    with model_col:
        st.subheader("Modell wählen")
        selected_model = st.radio(
            "Analyseweg",
            options=["Eigenes Modell", "YAMNet"],
            horizontal=True,
            label_visibility="collapsed",
        )
        model_key = "own" if selected_model == "Eigenes Modell" else "yamnet"
        status = model_status(model_key, PROJECT_ROOT)
        st.caption(status["message"])
        analyse_clicked = st.button("Analyse starten", type="primary", width="stretch")

    if analyse_clicked:
        if uploaded_file is None:
            st.warning("Bitte zuerst eine Audiodatei hochladen.")
            return
        with st.spinner("Audiodatei wird analysiert ..."):
            adapter = get_model_adapter(model_key, PROJECT_ROOT)
            try:
                result = adapter.predict(uploaded_file.getvalue(), uploaded_file.name)
            except Exception as exc:
                st.warning(f"Echter Modelllauf noch nicht verfügbar: {exc}")
                fallback_name = "Eigenes Modell" if model_key == "own" else "YAMNet"
                result = get_model_adapter("demo", PROJECT_ROOT).predict(
                    uploaded_file.getvalue(), uploaded_file.name
                )
                result = ModelResult(
                    label=result.label,
                    confidence=result.confidence,
                    model_name=f"{fallback_name} (Demo-Modus)",
                    is_demo=True,
                )
        st.write("")
        st.subheader("Analyseergebnis")
        render_warning(result)
        st.caption(f"Verwendeter Analyseweg: {result.model_name}")


def comparison_page() -> None:
    st.markdown('<div class="hb-kicker">Schulstunde 5</div>', unsafe_allow_html=True)
    st.header("Vergleich der Modellvarianten")
    st.write(
        "Beide Modelle werden mit demselben getrennten Testdatensatz geprüft. "
        "Die Tabelle ist vorbereitet und kann nach den Messungen ergänzt werden."
    )
    table = load_comparison_table(PROJECT_ROOT / "vergleich.csv")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Vergleichstabelle herunterladen",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="vergleich.csv",
        mime="text/csv",
    )


def training_page() -> None:
    st.markdown('<div class="hb-kicker">Schulstunde 2</div>', unsafe_allow_html=True)
    st.header("Eigenes Modell vorbereiten")
    st.write(
        "Hier wird kontrolliert, ob die vier Klassen für das eigene Teachable-Machine-Modell "
        "vollständig getrennt vorliegen. Der Testdatensatz bleibt vom Training getrennt."
    )

    status = dataset_status(PROJECT_ROOT)
    summaries = status["summaries"]
    table = pd.DataFrame(
        [
            {
                "Klasse": summary.key,
                "Training": summary.train_count,
                "Test": summary.test_count,
                "Trainingsstatus": "bereit" if summary.train_count >= 10 else "ergänzen",
                "Teststatus": "bereit" if summary.test_count >= 10 else "ergänzen",
            }
            for summary in summaries
        ]
    )
    st.dataframe(table, width="stretch", hide_index=True)

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Trainingsdateien", status["train_total"])
    metric_b.metric("Testdateien", status["test_total"])
    metric_c.metric("Exportstatus", "bereit" if model_status("own", PROJECT_ROOT)["ready"] else "offen")

    if status["ready_for_training"]:
        st.success("Alle vier Klassen haben mindestens zehn Trainingsdateien.")
    else:
        missing = ", ".join(status["missing_train"])
        st.warning(
            f"Für das Training fehlen noch mindestens zehn Dateien in: {missing or 'keiner Klasse'}."
        )

    metadata_path = PROJECT_ROOT / "data" / "metadata.csv"
    if metadata_path.exists():
        st.warning(
            "Der aktuelle Datensatz enthält synthetische Prototyp-Töne. "
            "Für die endgültige Bewertung müssen echte oder lizenzrechtlich "
            "geklärte Aufnahmen eingesetzt werden."
        )

    st.subheader("Ablauf für Schulstunde 2")
    st.markdown(
        """
        1. Trainingsdateien ausschließlich in `data/train/<klasse>/` ablegen.
        2. Testdateien ausschließlich in `data/test/<klasse>/` ablegen.
        3. In Teachable Machine ein Audioprojekt mit `alarm`, `siren`, `doorbell`
           und `background` anlegen.
        4. Nur die Trainingsdateien hochladen und das Modell trainieren.
        5. Als Keras-Modell exportieren und die Dateien nach `models/model_a/` legen.
        6. `keras_model.h5` und `labels.txt` kontrollieren und eine Alarmdatei testen.
        """
    )
    st.info(
        "Der Export ist erst vollständig, wenn `models/model_a/keras_model.h5` und "
        "`models/model_a/labels.txt` vorhanden sind."
    )


def about_page() -> None:
    st.markdown('<div class="hb-kicker">Projektkontext</div>', unsafe_allow_html=True)
    st.header("Über HörBlick")
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(
            """
            <div class="hb-panel">
            <h3>Problem und Zielgruppe</h3>
            <p>Viele Warnsignale im Alltag funktionieren ausschließlich über Schall.
            HörBlick soll diese Informationen sichtbar und verständlich machen.</p>
            <p>Die zentrale Persona ist Lena, 19, hochgradig schwerhörig und auf dem Weg
            in die erste eigene Wohnung.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="hb-panel">
            <h3>Grenzen und Quellen</h3>
            <p>Der Prototyp ist keine zertifizierte Sicherheitslösung und ersetzt keine
            normgeprüften Rauchmelder. Hintergrundgeräusche, Reichweite und Datenschutz
            beeinflussen die Erkennung.</p>
            <p>YAMNet: TensorFlow Hub, Apache-2.0. Eigene Modellvariante: Teachable Machine.
            Vollständige Quellen stehen in der README und Dokumentation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_styles()
    with st.sidebar:
        st.markdown("## HörBlick")
        st.caption("Material 3 Expressive inspiriertes Prüfungsprototyp")
        page = st.radio(
            "Navigation",
            ["Analyse", "Training", "Vergleich", "Über"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Status der ersten Schulstunde**")
        st.write("Projektstruktur angelegt")
        st.write("Datenordner vorbereitet")
        st.write("Streamlit-Grundgerüst aktiv")
        st.markdown("**Status der zweiten Schulstunde**")
        own_status = model_status("own", PROJECT_ROOT)
        st.write("Trainingsdaten geprüft")
        st.write("Eigenes Modell exportiert" if own_status["ready"] else "Export noch offen")

    if page == "Analyse":
        analyse_page()
    elif page == "Training":
        training_page()
    elif page == "Vergleich":
        comparison_page()
    else:
        about_page()

    st.markdown(
        '<div class="hb-footer">HörBlick ist ein Schulprojekt. Die Anzeige dient der Demonstration und ersetzt keine zertifizierte Warntechnik.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
