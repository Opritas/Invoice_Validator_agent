import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 1. SETUP ENVIRONMENT & API KEYS
# ==========================================
os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6Jz9Cf9lLKA4wE2ycxzPhZ1lPm2X0uiw9QrKok7PKKzLg" # Your active key
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "Invoice-Validator-Agent"

# Initialize Gemini 3.5 Flash
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

# ==========================================
# 2. DEFINE THE GRAPH STATE
# ==========================================
class InvoiceState(TypedDict):
    raw_invoice_text: str
    extracted_data: str
    validation_report: str
    is_valid: bool

# ==========================================
# 3. DEFINE THE PROCESSING NODES
# ==========================================
def extractor_node(state: InvoiceState) -> InvoiceState:
    print("\n--- [Node 1] Extracting Data from Invoice ---")
    
    system_prompt = (
        "You are an expert AI Data Extractor. Pull out the following fields clearly from the raw text:\n"
        "- Invoice Number\n- Date\n- Total Net Amount\n- Tax Amount\n- Total Amount Due\n"
        "- Seller VAT ID\n- Buyer VAT ID"
    )
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["raw_invoice_text"])
    ])
    
    # Clean text payload handling for Gemini structure
    extracted = response.content[0].get('text', '') if isinstance(response.content, list) else str(response.content)
    return {**state, "extracted_data": extracted}


def validator_node(state: InvoiceState) -> InvoiceState:
    print("\n--- [Node 2] Validating Extracted Invoice Metrics ---")
    
    system_prompt = (
        "You are an automated Compliance Auditor. Analyze the extracted invoice data.\n"
        "1. Check if the math is correct (Total Net + Tax = Total Amount Due).\n"
        "Note: Be smart—if a security deposit is listed, know it might be tax-exempt!\n"
        "2. Check if both the Seller and Buyer VAT IDs are present.\n"
        "3. Provide a clear text analysis.\n"
        "End your analysis with [VALID] if it passes all data rules, or [INVALID] if fields are broken or wrong."
    )
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["extracted_data"])
    ])
    
    report = response.content[0].get('text', '') if isinstance(response.content, list) else str(response.content)
    valid_flag = "[VALID]" in report
    
    print(f"Validation Result:\n{report.strip()}")
    return {**state, "validation_report": report, "is_valid": valid_flag}

# ==========================================
# 4. BUILD THE GRAPH FLOW
# ==========================================
builder = StateGraph(InvoiceState)

builder.add_node("extractor", extractor_node)
builder.add_node("validator", validator_node)

builder.add_edge(START, "extractor")
builder.add_edge("extractor", "validator")
builder.add_edge("validator", END)

app = builder.compile()

# ==========================================
# 5. EXECUTION ENTRYPOINT WITH INVOICE DATA
# ==========================================
if __name__ == "__main__":
    # Raw parsed layout data from Alpen_berens.pdf
    invoice_text = """
    Synthetic Invoice - Alpen Häuser Immobilien GmbH
    Rechnungsnummer: AH-2026-073 | Datum: 14.04.2026 | Währung: EUR
    Zekerbouw BV, Van Veldekestraat 4, 5671 VD Nuenen | VAT: NL823532549B01
    
    Line Items:
    1. Kaltmiete Büroeinheit: 2.100,00 €
    2. Nebenkosten-Vorauszahlung: 420,00 €
    3. Tiefgaragenstellplatz Nr. 42: 190,00 €
    4. Reinigungspauschale Gem.-Fl.: 85,50 €
    5. Mietkaution (Tax Exempt): 1.000,00 €
    
    Totals:
    Netto Gesamt: 3.795,50 €
    Umsatzsteuer (19% on taxable total): 531,15 €
    Gesamtbetrag: 4.326,65 €
    
    Seller VAT ID: DE812345678 | Due: 28.04.2026
    """
    
    initial_state = {
        "raw_invoice_text": invoice_text,
        "extracted_data": "",
        "validation_report": "",
        "is_valid": False
    }
    
    app.invoke(initial_state)