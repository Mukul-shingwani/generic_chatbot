from sys import flags
from openai import OpenAI
import os
import streamlit as st
import re
import requests
import pandas as pd
import yaml
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import tempfile
from st_audiorec import st_audiorec
# from audiorecorder import audiorecorder

client = OpenAI(api_key=st.secrets["openai"]["api_key"])

def build_prompt(user_query):
    return f"""
    You are an e-commerce shopping assistant based out of middle east in noon.com, majority customers are middle class.
    
    Your job:
    1. Detect if the query is about "planning" (like planning a party, picnic, etc.) or "shopping" (explicit buy orders) or "cooking/recipe".
    2. For planning queries
        - Start with a warm, brief **introductory line** summarizing the user's plan, e.g. "Sounds like you're planning a fun beach day! Here’s a list of essentials you might want to shop for:" but not just limited to this.
        - suggest a list of top 5 most relevant items, in order of relevance, that the user might want to buy online to fulfill the task. Be specific.
        - For example, instead of "return gifts", suggest things like "mini chocolates", "puzzle kits", "coloring books" etc.
        - Suggest items that make sense for the occasion and are typically bought online.
        - Only include **one specific item per search step** and keep it a bit relevant and concise not very long.
        - Donot apply unnecessary filters unless asked for 
    3. If intent is **shopping**:
    - Begin with a friendly confirmation, e.g.,  "Got it! You're looking to explore some great options for [product]. Here's a curated list of top brands you can check out:" but not just limited to this.
    - Keep the search step relevant and concise, focussing on the main category, not long or too specific.
    - Extract product/category name and optional filters like brand
    - If user uses vague brand indicators like:
        - "top brands", "luxury brands", "high-end" - Replace with **real premium brands actually found on Noon**.
        - "budget", "cheap", "affordable" - Replace with **real budget-tier brands actually found on Noon**.
    - NEVER hallucinate brands. Only include brands present on Noon.
    - Format all brand names in lowercase and underscores (e.g., tommy_hilfiger).
    - Return 7 most relevant brands when user asks for "top brands", "good brands" etc when exact brand names are provided to you use that only. If nothing around brands is mentioned donot apply.
    - Donot apply unnecessary filters unless asked for by the user, and take decision depending on search query/category.
    - Ensure the brands you are recommending are known and relevant as per the product category.
    Tier Examples :
        Luxury - YSL, Prada, Chanel, Gucci, Louis Vuitton
        Premium - Michael Kors, Coach, Guess, Tommy Hilfiger
        Mid-tier - Zara, Aldo, Nine West, Charles & Keith
        Budget - Caprese, Styli, Generic, Parfois, Duniso

    - STRICTLY enforce brand tier alignment:
        - If user asks for **budget** brands, ONLY return **budget-tier** brands (low-cost, value-driven).
        - If user asks for **top/luxury** brands, ONLY return **premium-tier** brands (high-end, designer, well-rated).
        - Do NOT mix tiers in a single result list.

    4. For **cooking/recipe** queries:
       - Begin with a natural suggestion, e.g.,  "Planning to cook butter chicken? Here's a quick list of items you’ll likely need and can easily order online:" but not just limited to this.
       - Identify the **top 5 essential ingredients or products** required for the recipe that a user can buy online.
       - Only suggest **non-perishable, e-commerce-friendly** items — i.e., things that are commonly sold online such as:
         - packaged spices (e.g., garam masala, turmeric, red chili powder)
         - cooking oils and ghee
         - ginger garlic paste
         - cooking cream, sauces, canned or frozen items (if relevant)
         - rice or packaged mixes (e.g., biryani mix, gravy base)
       - **Avoid** suggesting perishable items like fresh vegetables, milk, raw chicken, etc.
       - Think like an online grocery expert. Suggest items a user would likely need but may not already have at home.
       - Only 1 item per search step and keep it relevant and concise, focussing on the main ingredient, not long or too specific.
       - Do not give cooking instructions. Only extract shoppable items.
       - Donot apply unnecessary filters unless asked for 

    5. Output your answer in this format:
    <introductory message>
    intent: planning/shopping  
    search_steps:
    - {{q: "item1"}} or  
    - {{q: "item2", filters: {{brand: ["xyz","abc_123", "yyy", "z_rty", "pqrs", 'xd', 'yyu', '678', 'poi', 'lkj'], max_price: "100"}}}}

    6. Formatting rules for every q (apply to all intents)
    - Keep q minimal: 1 to 4 words, core product/category only relevant to ecommerce.
    - DO NOT add attributes (size, color, material, features, counts) unless the user explicitly asked for them.
    - Allowed extras in q: explicit user-provided brand (e.g., “nike”), explicit user-provided quantity/size like “1kg”, “500ml”, “128gb” when user asks for it.
    - NO parentheses, hyphens, slashes, or marketing adjectives in q.
    - Examples
        - ✅ "sunscreen", "beach towel", "power bank", "sugar"
        - ❌ "reef-safe sunscreen spf50 broad spectrum 200ml"
        - ✅ "1kg sugar" (only if user said 1kg)
        - ❌ "expandable 24-inch spinner suitcase with tsa lock", "low calorie powdered sugar"

    7. If the user query is something where you cannot help or its unethical/illegal but you can help them in some good way, then feel free to do so and give relevant search steps but donot keep it very long pls
    your introductory/caution message should be at max 15-20 words.
    
    Think like an e-commerce expert of middle east — only include things users can buy online, strictly relevant to ecommerce. Don’t mention services like booking a restaurant or sending invites.
    Be creative and conversational while forming the introductory message.
    
    Input: {user_query}
    Output:
    """

# Examples:
    
#     Input: "Help me plan a kids birthday party"
#     Output:
#     intent: planning
#     search_steps:
#     - {{q: "birthday balloons"}}
#     - {{q: "chocolate cake"}}
#     - {{q: "mini chocolates"}}
#     - {{q: "party snacks"}}
#     - {{q: "colorful paper plates"}}
    
#     Input: "Buy 1kg sugar of MDH under 100 aed, and 2kg tur dal from same brand"
#     Output:
#     intent: shopping
#     search_steps:
#     - {{q: "1kg sugar", filters: {{brand: "MDH", max_price: "100"}}}}
#     - {{q: "2kg tur dal", filters: {{brand: "MDH"}}}}

def get_search_plan(user_query):
    prompt = build_prompt(user_query)
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        # temperature=0.3
    )
    # response = client.responses.create(
    #                 model="gpt-4.1",
    #                 tools=[{"type": "web_search_preview"}],
    #                 input= prompt
    #             )
    return response.choices[0].message.content.strip()
    # response.output_text.strip()


def extract_queries(llm_text: str):
    """
    Try to parse the assistant's response for `search_steps`.
    - First attempt proper YAML.
    - If that fails, fall back to regex extraction of `q: "..."`.
    Returns a normalized list of dicts: [{'q': '...', 'filters': {...}}, ...]
    """
    steps = []

    # --- Try YAML first ---
    try:
        doc = yaml.safe_load(llm_text)
        if isinstance(doc, dict) and "search_steps" in doc:
            for item in doc.get("search_steps") or []:
                if isinstance(item, dict):
                    q = item.get("q") or item.get("query")
                    filters = item.get("filters") or {}
                elif isinstance(item, str):
                    q, filters = item.strip(), {}
                else:
                    continue

                if q:
                    # Normalize brand filter to list if a single string
                    if isinstance(filters, dict) and "brand" in filters and isinstance(filters["brand"], str):
                        filters["brand"] = [filters["brand"]]
                    steps.append({"q": q, "filters": filters})
            if steps:
                return steps
    except Exception as e:
        # Show parser problem but keep going with regex fallback
        with st.expander("⚠️ Failed to parse LLM response as YAML (show details)"):
            st.exception(e)

    # --- Regex fallback ---
    # Pattern 1: - {q: "item", ...}
    for m in re.findall(r'-\s*\{[^}]*q:\s*"([^"]+)"[^}]*\}', llm_text):
        steps.append({"q": m.strip(), "filters": {}})

    # Pattern 2: - q: "item"
    if not steps:
        for m in re.findall(r'-\s*q:\s*"([^"]+)"', llm_text):
            steps.append({"q": m.strip(), "filters": {}})

    # Pattern 3: bare q: "item" anywhere
    if not steps:
        for m in re.findall(r'\bq:\s*"([^"]+)"', llm_text):
            steps.append({"q": m.strip(), "filters": {}})

    return steps
    

# def extract_queries(llm_text):
#     try:
#         parsed = yaml.safe_load(llm_text)
#         if not parsed or "search_steps" not in parsed:
#             return []
#         return parsed["search_steps"]
#     except yaml.YAMLError as e:
#         st.error("⚠️ Failed to parse LLM response as YAML.")
#         st.exception(e)
#         return []


def show_product_carousel(df):
    html = '<div style="display: flex; overflow-x: auto; padding: 10px; width: 100%">'
    for _, row in df.iterrows():
        html += f'''
        <div style="flex: 0 0 auto; text-align: center; margin-right: 20px;">
            <a href="{row['Product URL']}" target="_blank">
                <img src="{row['Image URL']}" width="150" style="border-radius: 8px;">
            </a>
            <div style="font-weight:bold; margin-top:5px;">{row["Name"][:40]}...</div>
            <div>{row["Brand"]}</div>
            <div>AED {row.get("Sale Price (AED)", row.get("Price (AED)", "N/A"))}</div>
            <div>⭐ {row["Rating"]}</div>
        </div>
        '''
    html += '</div>'
    return html  # Return string, not IPython HTML



def fetch_top_products(query, country_code="AE", limit=3, sort_by="popularity", sort_dir="desc"):
    url = "https://api-app.noon.com/_svc/catalog/api/v3/search"

    params = {
        "q": query,
        "country": country_code,
        "limit": limit,
        "page": 1,
        "sort[by]": sort_by,
        "sort[dir]": sort_dir
    }

    headers = {
    "authority" : "api-app.noon.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Referer": "https://www.noon.com/",
    "Origin": "https://www.noon.com",
    "scheme" : "https",
    # "Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    # "Accept-Encoding": "gzip, deflate, br, zstd",
    # "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Cookie": 'visitor_id=1377ce41-a7be-4f7f-b968-010a37a1a5f2; _ym_uid=1720677223793794666; _fbp=fb.1.1721023114914.549776813895348518; _scid=5ea4c34e-b791-4c17-9f71-1d9b57c9de4f; GCP_IAP_UID=113324949211713157026; dcae=1; _tt_enable_cookie=1; th_capi_em=5c5a3c21a9b1551f0ec8d02bd7edc8b1167de12e937a99745d3f6f81c1e7a762; _nrtnetid="nav1.public.eyJ1c2VySWQiOiI1ZDYyOTgwZS0yNjNjLTRjNTUtYjA4MC04MmVmODJlOWYyZTAiLCJzaWQiOiI1MDBiYjE4MC04YzhmLTQwMGMtOTk1MS04YmM4NTBhZWZhZGMiLCJ0aWQiOiI0NTU3OWMxYzQ4MGU5N2I4OGJkMTVkZGYyNzFlZDBmNTYzNjAxYjM5NzQwMmU1ODc3ZGQ2YzA1NWQ0ZGM5NGJkIiwiaWF0IjoxNzMwNzI3MzkwfVg2OENkZWUzTHJzVGR3MUpGK1FIRVllSDlDUVZ3L0lYZkR1US9FakdBYTE4ei9NZUFBaHVEUkQrNTF6YWhicTRBME5Hek9tSC9aalVzWUo4dEpxNVV3dU0zN0xqaExoV0xKZFZheWl1YzRBa0x0SDN2K29JN2IzazFwSzI3ejR6RU1wMVUwbWpvNE1lQ1M1Q2wrL3J6YWdKNjdFU05FeXVuZ0RNbndsTlg2MTg2VDkrdVpXbkpDaTcwaVpMV0VBZXg4V0RjTGNGN29CWFhZbDg4TUdhRUovVi9XOElnUENSdy93aU1ZZFhEbDF1ZlE4R2VmOUx2dkt4WVRFdFVUK3kvUU1CekVTMjlVVUhFR0NKOVZTTFFITkFFOE51V1YxQWVvZng1VTdsbWlHR0RFczllWE5JVzlpNXQxRnZHT3B6Rjh4TGNldTZTUG52L2E3dTNIdm5lc2Z6eVF1clVFYis1RGg0SGdzYllESjFxRWhZSGFSd1Y2Rm16Mk9aVTNBVEVzVm1uRWZMUGtPV1NHcE1OdkNaZUJNS1UwN2xBZHRSVXk3N05vdi9acThKOWtxMEgyUHgrK3hNY0JQVzBqZ0tHU0xpbEpDWGtKbGlJUEFoZjhCWHp0bnNmUks2ZUF0OHdVMCs1QmcvK0FnaEd6eVFHWGZaWjdwT2gzZ241V3Bz.MQ=="; _ttp=JZOnddQZGd7KZPXfIQkc20qU1Jr.tt.1; _pin_unauth=dWlkPVpXSXhPVGcwWkRVdFlqWXhZUzAwWmpsakxUZzRaV1l0TlRJNU9EY3lOakpsTnpkaA; supported-image-formats={"avif":true,"webp":true}; _ga_43G6NV0HZY=GS1.1.1739187627.2.0.1739187631.0.0.0; ph_phc_qNKORfyT0LoPjVeJTJ8FfAhCnpzgGBSkZmT27spzR23_posthog=%7B%22distinct_id%22%3A%2201958983-32b6-7dfd-9ecd-39c5316e92b4%22%2C%22%24sesid%22%3A%5B1742539730936%2C%220195b775-f896-7626-87c3-ecbc744cedb3%22%2C1742539716758%5D%2C%22%24initial_person_info%22%3A%7B%22r%22%3A%22%24direct%22%2C%22u%22%3A%22https%3A%2F%2Fwww.noon.com%2Fuae-en%2Fprue-puff-sleeve-tiered-dress-blue%2FN49961258V%2Fp%2F%3Futm_source%3DC1000094L%26utm_medium%3Dreferral%22%7D%7D; th_capi_db=ce7653edd037e816d7412eac845cb13026002c2cc3b34129bc8399d5bad6f6a5; __gads=ID=baef72be5c7f55d6:T=1723055350:RT=1745507491:S=ALNI_MbENyLCU-F52CzjsRtvzENDS5-Uqw; __gpi=UID=00000eb8c37ab559:T=1723055350:RT=1745507491:S=ALNI_MYcaevQRQ-SFOEkpmNSYnr-02jK9A; __eoi=ID=bf2f059439d0fed3:T=1744120524:RT=1745507491:S=AA-AfjZPLdv0ydXoS1ofnqnWpxBh; _gcl_gs=2.1.k1$i1747903570$u51244222; ttcsid_CMSCRUBC77U72P15LFN0=1747901962426::l_k9Ymo42jlpk6KM6U6G.2.1747903599280; _ga_MTC4V6QW17=GS2.1.s1747901961$o16$g1$t1747903611$j20$l0$h0$dxK__u9Iaf9AirYYUJsavL3gC-hQii0abgw; _ga=GA1.2.498398072.1720677220; _abck=A8E42B1082197CCBC98A86A6BA70F59B~0~YAAQPvEBF+zpzy2XAQAAk8aYOg4C61SxwI3ZvztQ1xcuW1CeLALk579TjlUIBiWrk5ZILr9tdRSAOK63HOjf9aoQRixzCrUAkCl5fRPj/Fbrfi7lb8p+f4Ps88s598n99BdWrD7ikCb3Y9DkJIOY7vV45LeEHr27H9oCbJ1CL5cX9GEjvASVej9tUT0Nx265GDyL2F57tnFenYx97p8sZVrQyxP7i02Yb1uzmGg4IanvXRvcupD9jpIXLjNApEf1M1mizdW24SATw9Ue6jECOXdXpjlgf9ZgwFQfLLIaKIMuTJJjcvsIib0yS1pm6v24JlLBEV3dsipst/1LyyacfZ3jKSmSGcfDTIUk0BOVOx4dEug3DTk5nRWNRWhc4/SwWa4gr8MyZ7QshaCl830KSG2tQYOWiopgMfOyXSlWWMJLI9VcduEruXIKu54A4KuLkvYL4EZsTK4cOkUSuFWJQSLc4kVLMQFYsBN0YgjC0aEiuwBnu7CSgMeRmbedStETZD3iEVILSR0z4mYf/P3E~-1~-1~-1; _gcl_aw=GCL.1749631128.CjwKCAiAxKy5BhBbEiwAYiW--56APYmycZBFUZsr5uo_0aBYX9dQ7dks53azpAkJOf0IrEhHUz_J6hoCmTYQAvD_BwE; review_lang=xx; x-location-ecom-ae=eyJsYXQiOiAyNTE4NDEzOTQsICJsbmciOiA1NTI2MTUwODAsICJhZGRyZXNzX2NvZGUiOiAiNGQ4YjE3NTdiMDkxMGJjMzBkYTg4MmZkYmQyNGM1YjMiLCAiYXJlYSI6ICJNYXJhc2kgRHIgLSBCdXNpbmVzcyBCYXkgLSBEdWJhaSAtIER1YmFpIn0; th_capi_fn=4422e151fd18e8ff239a9b97e5ea80e26286cc7ce04cef2cfb3ecab63743216d; ttcsid_CJCRUKJC77U5K7SPETSG=1750924595230::HDqe-IJnl42tR9R3zgr-.1.1750924878285; _gcl_au=1.1.649792036.1751896820; _ym_d=1752475306; nloc=en-ae; _sctr=1%7C1754245800000; ZLD887450000000002180avuid=cd3ce55c-a7ef-4f54-bc9c-ea036951ebdf; x-whoami-headers=eyJ4LWxhdCI6IjI1MTg0MTM5NCIsIngtbG5nIjoiNTUyNjE1MDgwIiwieC1hYnkiOiJ7XCJpcGxfZW50cnlwb2ludC5lbmFibGVkXCI6MSxcIndlYl9wbHBfcGRwX3JldmFtcC5lbmFibGVkXCI6MSxcImNhdGVnb3J5X2Jlc3Rfc2VsbGVyLmVuYWJsZWRcIjoxfSIsIngtZWNvbS16b25lY29kZSI6IkFFX0RYQi1TNSIsIngtbm9vbmluc3RhbnQtem9uZWNvZGUiOiJXMDAxMDYzMDdBIiwieC1hYi10ZXN0IjpbNjEsOTQxLDk2MSwxMDMxLDEwODEsMTA5MCwxMTAxLDExNjIsMTIxMSwxMjUxLDEyOTEsMTMwMSwxMzMxLDEzNjIsMTM3MSwxNDEzLDE0MjEsMTQ1MCwxNDcxLDE1MDIsMTU0MSwxNTgwLDE2MjEsMTY1MCwxNzAxLDE3MjEsMTc1MCwxODExXSwieC1yb2NrZXQtem9uZWNvZGUiOiJXMDAwNjg3NjVBIiwieC1yb2NrZXQtZW5hYmxlZCI6dHJ1ZSwieC1pbnRlcm5hbC11c2VyIjp0cnVlLCJ4LWJvcmRlci1lbmFibGVkIjp0cnVlfQ%3D%3D; nguestv2=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJraWQiOiI4ZGNkNTMwMmRlODI0M2VkOWIwMjBhMmVjZmZlZDE0NiIsImlhdCI6MTc1NDM4NTAwNywiZXhwIjoxNzU0Mzg1MzA3fQ.x9nBB_-zRIqpU-fOKgw8AQc6g1weZ8T_j1EsfCStkj4; AKA_A2=A; _etc=qFEqWRV23rXo9QVu; _uetsid=b0340d9070ff11f09f2c57faffb66605; _uetvid=ec7f92403f4911ef9f1ae5159765f4c9; _scid_r=C7pepMNOt5EeF8txHZtXyd5PGi2VUhGJ2INlNg; _clck=u3wzwt%7C2%7Cfy7%7C0%7C1653; _ScCbts=%5B%22334%3Bchrome.2%3A2%3A5%22%2C%22385%3Bchrome.2%3A2%3A5%22%5D; ttcsid_CFED02JC77U7HEM9PC8G=1754388029450::nEvSFYVU6L7wC0-i4lK7.42.1754389961481; ttcsid=1754388029450::xsdRlN-FRnpCNGUiR-qc.44.1754389961481; _clsk=pj9qkj%7C1754389962043%7C22%7C0%7Cz.clarity.ms%2Fcollect; RT="z=1&dm=noon.com&si=a684c35f-702c-4ffc-836c-55ae623b8e10&ss=mdybj9kj&sl=0&tt=0&rl=1"; _natnetidv2=eyJhbGciOiJLTVNSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdGlkIjoiNDU1NzljMWM0ODBlOTdiODhiZDE1ZGRmMjcxZWQwZjU2MzYwMWIzOTc0MDJlNTg3N2RkNmMwNTVkNGRjOTRiZCIsInNpZCI6IjUwMGJiMTgwLThjOGYtNDAwYy05OTUxLThiYzg1MGFlZmFkYyIsInd0aWQiOiI0NmFhNmRmMy0wMDg5LTQxYzItYTJjOS1kN2ZjYWVjYTUyMDEiLCJpYXQiOjE3NTQzOTExOTYsImV4cCI6MTc1NDM5MTQ5Nn0.0W0gxCu8w5KwYHUCVHs-g9BMayZ-RIT-N-Qv1-Hz7e_sYmJBO9Tt95PQRhchLzU92H735NRMtwCFsFe2jlMylY3BkX3KQuvpNrXOBxvPtQMoYHCpG-P9Y3aTj9xoi9cWmJOaubO3YU72g291mVGghWpPMseVzgRCCk6smzbWM0cSx3EAcfogt4VsWHI1z0XvGrFvSa8cdXF60B6ws61OFJY0E0nk0hFfwIpfzB_6uDi70kc4fi5kc88W3m2bS7sgDbxFTajbrzzehx9pQe2wtMKAkGfBHWUXCkcAWTSq1QT4M4V9TrM7IAP1x4OjMTthn4Osw0IezjUppjwmX2SKyil96k1i_A4kuf-EQ2-tpYHWOaqfg7ODJ_1yZcD1AjeQk-SxSVP4p6t1pEjkkF-dSjymKnAudrsTQG_z_kmarjwfB8qvVWmARRkcjyPidPeKaUl0lJQvsdeUFV2P6mUy3BMwowEGVFgyLvsptsGgT4aPlG_PLuXNQChm70AOY-sp',
    "Sec-Ch-Ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
    "Sec-Ch-Ua-Mobile": "?0",
    # "Sec-Ch-Ua-Platform": "macOS",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests" : "1"
    }

    try:
        response = requests.get(url, params=params, headers=headers)

        # st.write(f"🔗 URL: {response.url}")
        # st.write(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            st.error(f"❌ Failed to fetch products. Status code: {response.status_code}")
            return pd.DataFrame()

        if not response.text.strip():
            st.error("❌ Empty response body. API may be blocking deployed server.")
            return pd.DataFrame()

        try:
            data = response.json()
        except Exception as json_err:
            st.error("❌ Response is not valid JSON:")
            st.text(response.text[:1000])  # Show first 1k characters
            return pd.DataFrame()

        products = data.get("hits", [])[:limit]

        if not products:
            st.warning(f"No products returned for query: {query}")
            return pd.DataFrame()

        results = []
        for product in products:
            image_key = product.get("image_key")
            image_url = f"https://f.nooncdn.com/p/{image_key}.jpg?width=800" if image_key else "N/A"

            results.append({
                "SKU": product.get("sku", "N/A"),
                "SKU Config": product.get("sku_config", "N/A"),
                "Name": product.get("name", "N/A"),
                "Brand": product.get("brand", "N/A"),
                "Image URL": image_url,
                "Price (AED)": product.get("price", "N/A"),
                "Sale Price (AED)": product.get("sale_price", "N/A"),
                "Rating": product.get("product_rating", {}).get("value", "N/A"),
                "Product URL": f"https://www.noon.com/uae-en/{product.get('sku', '')}/p/"
            })

        return pd.DataFrame(results)

    except Exception as e:
        st.exception(f"❌ Exception while fetching products for query: {query}\n{e}")
        return pd.DataFrame()


def build_batched_validation_prompt(user_query, df):
    header = (
        "You validate product relevance for noon.com. basis the user query and help take a decision to recommend the relevant products.\n"
        f'User Query: "{user_query}"\n\n'
        "You need to Decide accurately if each product matches the Search Step and is relevant to the User Query. Treat the search step as a sub step in fulfilling the original user query to get more context.\n"
        "- Mark 1 if it's the same item/category or a very close match that makes sense to be shown to the user as per their query.\n"
        "- Mark 0 if it's a different category, off-topic, or a combo that changes the core ingredient or essence (e.g., 'tomato & mascarpone' ≠ 'mascarpone cheese'). But donot be over harsh here, if the product is still somewhat related, given it will not be highly illogical to suggest to user, consider giving it a 1.\n"
        "- Focus more on the main user query, if the product is relevant to the main user query and not highly irrelevant to the search step, then mark it as 1.\n"
        "- Be a little stricter for cooking/ingredient-like steps (spices, oil, ghee, cocoa, etc.).\n"
        "- Consider yourself to be the user, and think if you would be happy to see this product in the results for the things you asked.\n\n"
        "Output format (IMPORTANT): return ONLY one line per product in the SAME ORDER as given:\n"
        "<SKU> : 0 or 1\n"
        "No extra text.\n\n"
        "Evaluate these products:\n"
    )

    lines = []
    for _, row in df.iterrows():
        lines.append(
            f'Search Step: "{row["search_step"]}"\n'
            f'Product Name: "{row["Name"]}"\n'
            f'SKU: "{row["SKU"]}"\n'
        )
    return header + "\n".join(lines) + "\nReturn only the lines: <SKU> : 0/1\n"


def _parse_sku_flags(text):
    mapping = {}
    for line in text.splitlines():
        line = line.strip().strip(",").replace('"', '').replace("'", "")
        if ":" not in line:
            continue
        sku, val = line.split(":", 1)
        sku = sku.strip()
        # pick first 0/1 digit in the value
        m = re.search(r"[01]", val)
        if sku and m:
            mapping[sku] = int(m.group(0))
    return mapping


def validator_llm_batched(user_query, df):
    prompt = build_batched_validation_prompt(user_query, df)

    with st.expander("🔍 Prompt Sent to Validator"):
        st.code(prompt, language="markdown")

    try:
        resp = client.chat.completions.create(
            model="gpt-5-nano", 
            messages=[{"role": "user", "content": prompt}],
            # temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()

        with st.expander("📬 Validator Response (SKU : 0/1)"):
            st.code(content, language="text")

        mapping = _parse_sku_flags(content)
        # Default to 0 for any missing SKU to stay safe
        return {sku: mapping.get(sku, 0) for sku in df["SKU"].tolist()}

    except Exception as e:
        st.error(f"❌ Validation failed: {e}")
        return {sku: 0 for sku in df["SKU"].tolist()}



# Optional local transcription (free) via faster-whisper
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except Exception:
    HAS_WHISPER = False

if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None
    if HAS_WHISPER:
        try:
            # int8 keeps it light on CPU-only boxes
            st.session_state.whisper_model = WhisperModel("base", compute_type="int8")
        except Exception as e:
            st.warning(f"Could not load Whisper locally: {e}")

import tempfile, os

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Persist WAV bytes to a tmp file -> faster-whisper -> text."""
    if not HAS_WHISPER or st.session_state.whisper_model is None:
        st.info("Transcription model not available on this server.")
        return ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        wav_path = tmp.name
    try:
        segments, _ = st.session_state.whisper_model.transcribe(wav_path)
        return " ".join(seg.text for seg in segments).strip()
    finally:
        try: os.unlink(wav_path)
        except Exception: pass


# ================== PAGE & STATE ==================
st.set_page_config(page_title="noon Assistant", layout="wide")
st.title("🛍️ noon Assistant")
st.markdown("Type your query or use voice. We’ll generate a plan, fetch products, and validate relevance.")

for key, default in [
    ("text", ""),
    ("last_audio_len", 0),   # to avoid re-transcribing the same audio on reruns
]:
    if key not in st.session_state:
        st.session_state[key] = default


def run_pipeline(user_query: str):
    # 1) Get plan + show raw text
    with st.spinner("🤖 Generating search plan using GenAI..."):
        try:
            result = get_search_plan(user_query)
        except Exception as e:
            st.error("Failed to generate a search plan.")
            with st.expander("Details"):
                st.exception(e)
            return

        st.markdown("### ✨ Detected Search Steps")
        st.code(result, language="yaml")

    # 2) Extract queries safely
    queries = []
    try:
        queries = extract_queries(result)
    except Exception as e:
        with st.expander("⚠️ Couldn’t extract search steps (show details)"):
            st.exception(e)

    if not queries:
        st.warning("I couldn’t extract valid search steps from the model’s reply. I’ll try your original query directly.")
        queries = [{"q": user_query, "filters": {}}]

    # 3) Fetch products
    results = []
    with st.spinner("⏳ Hang on, getting the best recommendations for you..."):
        try:
            for q_step in queries:
                # supports dict {q, filters} or plain string
                if isinstance(q_step, dict):
                    q = q_step.get("q")
                    filters = q_step.get("filters", {}) or {}
                else:
                    q, filters = str(q_step), {}

                if not q:
                    continue

                if "brand" in filters and isinstance(filters["brand"], (list, tuple)):
                    for brand in filters["brand"]:
                        df_item = fetch_top_products(query=f"{q}/{brand}")
                        if not df_item.empty:
                            df_item["search_step"] = q
                            results.append(df_item)
                else:
                    df_item = fetch_top_products(query=q)
                    if not df_item.empty:
                        df_item["search_step"] = q
                        results.append(df_item)
        except Exception as e:
            st.error("There was a problem while fetching products.")
            with st.expander("Details"):
                st.exception(e)
            return

    if not results:
        st.warning("No products found. Try refining your query.")
        return

    # 4) Validate relevance (but don’t crash if validator fails)
    import pandas as pd
    df = pd.concat(results, ignore_index=True)

    with st.spinner("🔍 Validating product relevance..."):
        try:
            sku_to_flag = validator_llm_batched(user_query, df)  # expected {sku: 0/1}
            if isinstance(sku_to_flag, dict):
                df["is_relevant"] = df["SKU"].map(sku_to_flag).fillna(0).astype(int)
            else:
                # If validator returns something unexpected, don’t drop everything
                st.info("Validator returned an unexpected format; showing unfiltered results.")
                df["is_relevant"] = 1
        except Exception as e:
            st.info("Validator unavailable; showing unfiltered results.")
            with st.expander("Validator error (details)"):
                st.exception(e)
            df["is_relevant"] = 1

    df = df[df["is_relevant"] == 1]

    if df.empty:
        st.warning("No relevant products found after validation.")
        return

    # 5) Render carousel
    st.markdown("### 🛒 Top Product Recommendations")
    try:
        html = show_product_carousel(df)
        try:
            st.html(html)
        except Exception:
            st.components.v1.html(html, height=420, scrolling=True)
    except Exception as e:
        st.error("Failed to render product carousel.")
        with st.expander("Details"):
            st.exception(e)



# # ================== PIPELINE (full-width) ==================
# def run_pipeline(user_query: str):
#     # Everything below renders full-width (no columns)
#     with st.spinner("🤖 Generating search plan using GenAI..."):
#         result = get_search_plan(user_query)
#         queries = extract_queries(result)
#         st.markdown("### ✨ Detected Search Steps")
#         st.code(result, language="yaml")

#     results = []
#     with st.spinner("⏳ Hang on, getting the best recommendations for you..."):
#         for q_step in queries:
#             # supports dict {q, filters} or plain string
#             if isinstance(q_step, dict):
#                 q = q_step.get("q")
#                 filters = q_step.get("filters", {}) or {}
#             else:
#                 q, filters = str(q_step), {}

#             if not q:
#                 continue

#             if "brand" in filters and isinstance(filters["brand"], (list, tuple)):
#                 for brand in filters["brand"]:
#                     df_item = fetch_top_products(query=f"{q}/{brand}")
#                     if not df_item.empty:
#                         df_item["search_step"] = q
#                         results.append(df_item)
#             else:
#                 df_item = fetch_top_products(query=q)
#                 if not df_item.empty:
#                     df_item["search_step"] = q
#                     results.append(df_item)

#     if not results:
#         st.warning("No products found. Try refining your query.")
#         return

#     import pandas as pd
#     df = pd.concat(results, ignore_index=True)

#     with st.spinner("🔍 Validating product relevance..."):
#         # returns {sku: 0/1}
#         sku_to_flag = validator_llm_batched(user_query, df)
#         df["is_relevant"] = df["SKU"].map(sku_to_flag).fillna(0).astype(int)
#         df = df[df["is_relevant"] == 1]

#     if df.empty:
#         st.warning("No relevant products found after validation.")
#         return

#     st.markdown("### 🛒 Top Product Recommendations")
#     html = show_product_carousel(df)
#     try:
#         st.html(html)
#     except Exception:
#         st.components.v1.html(html, height=420, scrolling=True)


# ================== INPUTS (text + voice) ==================
# Text box spans full width and reruns every keystroke so the Generate button enables immediately.
user_box_val = st.text_input(
    "💬 Your query (type or record):",
    value=st.session_state.text,
    key="query_box",
    placeholder="e.g., Help me plan a beach picnic",
)
if user_box_val != st.session_state.text:
    st.session_state.text = user_box_val

# Voice recorder block — this widget draws its own buttons.
st.caption("🧑‍🎤 Or use your voice below:")
audio_bytes = st_audiorec()
# audio_bytes = audiorecorder("Click to record", "Click to stop recording")

# If user recorded something new, transcribe once and sync to the text box
if audio_bytes and len(audio_bytes) != st.session_state.last_audio_len:
    st.session_state.last_audio_len = len(audio_bytes)
    with st.spinner("🎙️ Transcribing audio..."):
        transcript = transcribe_audio_bytes(audio_bytes)
    if transcript:
        # st.success("✅ Transcribed. You can edit the text box above, or hit Generate.")
        st.session_state.text = transcript
        st.rerun()  # refresh to reflect transcribed text in the box instantly


# ================== ACTION ROW (right-aligned buttons only) ==================
# Wide spacer + tight right controls; keeps results that follow in full-width.
wrap_l, wrap_mid, wrap_r = st.columns([1, 2, 1])

with wrap_mid:
    can_generate = bool(st.session_state.text.strip())
    b1, b2 = st.columns([1, 1])
    with b1:
        generate_clicked = st.button("✨ Generate", use_container_width=True, disabled=not can_generate)
    with b2:
        clear_clicked = st.button("🧹 Clear", use_container_width=True)

if clear_clicked:
    st.session_state.text = ""
    st.session_state.last_audio_len = 0
    st.rerun()

st.markdown("---")

# ================== RUN (full width) ==================
if generate_clicked:
    user_query = st.session_state.text.strip()
    st.markdown("#### Working on it…")
    st.write(user_query)
    run_pipeline(user_query)


# st.set_page_config(page_title="noon Assistant", layout="wide")

# st.title("🛍️ noon Assistant")
# st.markdown("Enter your query — whether it's a **plan**, a **buying task**, or **recipe support**, and we’ll fetch the top picks!")

# user_query = st.text_input("💬 What do you need help with?", placeholder="e.g., Help me plan a beach picnic", key="user_query")

# if st.button("Generate Search Plan & Show Products") and user_query:
#     with st.spinner("🤖 Generating search plan using GenAI..."):
#         result = get_search_plan(user_query)
#         queries = extract_queries(result)
#         st.markdown("#### ✨ Detected Search Steps")
#         st.code(result, language="yaml")

#     results = []

#     with st.spinner("⏳ Hang on, getting the best recommendations for you..."):
#         for i, q_step in enumerate(queries):
#             q = q_step.get("q")
#             filters = q_step.get("filters", {})

#             if not q:
#                 continue

#             if "brand" in filters:
#                 for brand in filters["brand"]:
#                     brand_query = f"{q}/{brand}"
#                     df_item = fetch_top_products(query=brand_query)
#                     if not df_item.empty:
#                         df_item["search_step"] = q
#                         results.append(df_item)
#             else:
#                 df_item = fetch_top_products(query=q)
#                 if not df_item.empty:
#                     df_item["search_step"] = q
#                     results.append(df_item)

#     if results:
#         df = pd.concat(results, ignore_index=True)

#         with st.spinner("🔍 Validating product relevance..."):
#             sku_to_flag = validator_llm_batched(user_query, df)
#             df["is_relevant"] = df["SKU"].map(sku_to_flag).fillna(0).astype(int)
#             df = df[df["is_relevant"] == 1]
#             # st.code(df)
#         if df.empty:
#             st.warning("No relevant products found after validation.")
#         else:
#             st.markdown("#### 🛒 Top Product Recommendations")
#             html_carousel = show_product_carousel(df)
#             st.html(html_carousel)
#     else:
#         st.warning("No products found. Try refining your query.")
