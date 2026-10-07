import streamlit as st

from pipeline import ask, load_resources, years_by_ticker

st.set_page_config(page_title="10-K Assistant", layout="wide")


@st.cache_resource
def get_resources():
    return load_resources()


with st.spinner("Loading models and index (first run only)..."):
    resources = get_resources()

years_for = years_by_ticker(resources["chunks"])
tickers = list(years_for)
methods = list(resources["retrievers"])

st.title("SEC 10-K Assistant")
st.caption("Answers are grounded in the retrieved excerpts, with citations.")

question = st.text_input(
    "Question", placeholder="What are Apple's net sales by product category?"
)

with st.expander("Settings", expanded=True):
    col1, col2, col3 = st.columns(3)

    ticker = col1.selectbox("Company", ["All"] + tickers)

    if ticker == "All":
        year_options = sorted({y for ys in years_for.values() for y in ys})
        default = 0                    # "All" years
    else:
        year_options = years_for[ticker]
        default = len(year_options)    # latest filing (All is at 0)

    year = col2.selectbox(
        "Fiscal year", ["All"] + year_options, index=default, key=f"year_{ticker}"
    )
    method = col3.selectbox("Retrieval method", methods, index=methods.index("bm25"))
    k = st.slider("Chunks to retrieve (k)", 3, 10, 5)
    st.caption(
        "BM25 + filter is the default. bm25+rerank was best on the eval but slower. "
        "Hybrid is weighted 1:1."
    )

asked = st.button("Ask")

if asked and question.strip():
    try:
        with st.spinner("Retrieving and answering..."):
            out = ask(
                question,
                resources,
                method=method,
                ticker=None if ticker == "All" else ticker,
                fiscal_year=None if year == "All" else year,  # keep as string
                k=k,
            )
    except Exception as e:
        st.error(f"Something went wrong: {e}")
    else:
        st.subheader("Answer")
        st.write(out["answer"].replace("$", r"\$"))

        with st.expander(f"Retrieved chunks ({len(out['chunks'])})"):
            for i, c in enumerate(out["chunks"], start=1):
                st.markdown(f"**{i}. {c['label']}**")
                st.text(c["text"])