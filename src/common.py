import re
from bs4 import BeautifulSoup
from sec_parser import TreeNode

SECTION_ITEMS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Security",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Saftey Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "MD&A",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits, Financial Statement Schedules",
}

# create regex
ITEM_HEADING_RE = re.compile(
    r"^\s*item\s+(\d{1,2}[a-c]?)\.?\s*[-—:]?\s*",
    re.IGNORECASE | re.MULTILINE,
)

def clean_text(s):
    s = s.replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", s).strip()

# return dictionary of item number : node
def get_item_nodes(tree) -> dict[str, TreeNode]:
    item_nodes = {}

    def dfs(node):
        el = node.semantic_element
        if type(el).__name__ == "TopSectionTitle":
            m = ITEM_HEADING_RE.match(el.text)
            if m:
                item_nodes[m.group(1).upper()] = node

        for c in node.children:
            dfs(c)

    for root in tree:
        dfs(root)
    
    return item_nodes

# get text for this node
def get_text(node) -> list[str]:
    text = [clean_text(node.semantic_element.text)]
    for c in node.children:
        text.extend(get_text(c))
    return text

def get_sections(tree) -> dict[str, str]:
    sections = {}

    item_nodes = get_item_nodes(tree)
    for item_num, node in item_nodes.items():
        text = "\n".join(get_text(node)).strip()
        if text:
            sections[item_num] = text
    
    return sections

def section_label(item_num: str) -> str:
    return SECTION_ITEMS.get(item_num, f"Item {item_num}")