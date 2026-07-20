import os
import re
import json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 1. SETUP ENVIRONMENT & API KEYS
# ==========================================
# Ollama runs locally - COMPLETELY FREE!
# Download from: https://ollama.ai
# Then run: ollama pull llama2 (or another model)
# Ollama will run on http://localhost:11434

try:
    # Ollama runs locally - choose any model you've pulled
    llm = OllamaLLM(model="llama2", temperature=0)
except Exception as e:
    print(f"Error initializing Ollama: {e}")
    print("\nPlease install Ollama:")
    print("1. Download from: https://ollama.ai")
    print("2. Run: ollama pull llama2 (or neural-chat, mistral, etc.)")
    print("3. Start Ollama server (it will run on localhost:11434)")
    print("4. Then run this script again")
    exit(1)

# ==========================================
# 2. DEFINE THE GRAPH STATE
# ==========================================
class InvoiceState(TypedDict):
    raw_invoice_text: str
    extracted_data: str
    validation_report: str
    tax_percentage: float
    is_valid: bool

# ==========================================
# 3. DEFINE THE PROCESSING NODES
# ==========================================
def extractor_node(state: InvoiceState) -> dict:
    """Extract invoice data using Ollama."""
    print("\n--- [Node 1] Extracting Data from Invoice ---")
    
    system_prompt = (
        "You are an expert AI Data Extractor. Extract the following fields from the invoice text.\n"
        "Respond in JSON format with these exact keys:\n"
        "- invoice_number\n"
        "- date\n"
        "- net_amount (in EUR, remove currency symbols)\n"
        "- tax_amount (in EUR)\n"
        "- total_due (in EUR)\n"
        "- seller_vat_id\n"
        "- buyer_vat_id\n"
        "- tax_percentage (calculate: tax_amount / net_amount * 100)\n"
        "If a field is not found, use null for that field."
    )
    
    try:
        # Combine system prompt and invoice text
        full_prompt = f"{system_prompt}\n\nInvoice:\n{state['raw_invoice_text']}"
        
        response = llm.invoke(full_prompt)
        
        # Extract text content from response
        extracted = response if isinstance(response, str) else str(response)
        print(f"Extracted data:\n{extracted}")
        
        return {
            **state,
            "extracted_data": extracted,
            "tax_percentage": 0.0
        }
    except Exception as e:
        print(f"Error in extractor node: {e}")
        return {
            **state,
            "extracted_data": f"Error: {str(e)}",
            "tax_percentage": 0.0
        }


def validator_node(state: InvoiceState) -> dict:
    """Validate extracted invoice data."""
    print("\n--- [Node 2] Validating Extracted Invoice Metrics ---")
    
    system_prompt = (
        "You are an automated Compliance Auditor. Analyze the extracted invoice data.\n"
        "Perform the following checks:\n"
        "1. Verify mathematical accuracy: Net Amount + Tax Amount approximately equals Total Due (allow 0.01 EUR rounding)\n"
        "2. Check if Tax Percentage is correctly calculated\n"
        "3. Verify both Seller VAT ID and Buyer VAT ID are present and properly formatted\n"
        "4. Flag any missing or invalid fields\n"
        "5. Note if a security deposit is marked as tax-exempt\n\n"
        "Provide a detailed analysis and end with:\n"
        "[VALID] if all checks pass, or [INVALID] if any critical issues are found."
    )
    
    try:
        # Combine system prompt and extracted data
        full_prompt = f"{system_prompt}\n\nExtracted Data:\n{state['extracted_data']}"
        
        response = llm.invoke(full_prompt)
        
        report = response if isinstance(response, str) else str(response)
        valid_flag = "[VALID]" in report
        
        # Try to extract tax percentage from the extracted data
        tax_percentage = 0.0
        try:
            # Look for JSON in extracted data
            json_match = re.search(r'\{.*\}', state["extracted_data"], re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "tax_percentage" in data and data["tax_percentage"] is not None:
                    tax_percentage = float(data["tax_percentage"])
        except (json.JSONDecodeError, ValueError):
            pass
        
        print(f"Validation Report:\n{report.strip()}")
        
        return {
            **state,
            "validation_report": report,
            "is_valid": valid_flag,
            "tax_percentage": tax_percentage
        }
    except Exception as e:
        error_msg = str(e).encode('utf-8', errors='ignore').decode('utf-8')
        print(f"Error in validator node: {error_msg}")
        return {
            **state,
            "validation_report": f"Error: {str(e)}",
            "is_valid": False,
            "tax_percentage": 0.0
        }

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
    # Raw invoice data example
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
    
    Seller VAT ID: DE812345678 | Buyer VAT ID: NL823532549B01 | Due: 28.04.2026
    """
    
    initial_state = {
        "raw_invoice_text": invoice_text,
        "extracted_data": "",
        "validation_report": "",
        "tax_percentage": 0.0,
        "is_valid": False
    }
    
    print("=" * 60)
    print("INVOICE VALIDATOR AGENT - Powered by Ollama & LangGraph")
    print("=" * 60)
    
    result = app.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    print(f"[+] Invoice Valid: {result['is_valid']}")
    print(f"[+] Tax Percentage: {result['tax_percentage']:.2f}%" if result['tax_percentage'] else "[+] Tax Percentage: Not extracted")
    print("\nValidation Report:")
    print(result["validation_report"])