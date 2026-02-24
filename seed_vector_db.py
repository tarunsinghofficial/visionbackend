"""Seed ChromaDB with furniture product data.

Run once:  python seed_vector_db.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_store"
COLLECTION_NAME = "furniture_products"

FURNITURE_DATA = [
    # Living Room
    {"id": "prod_001", "name": "Modern Velvet Sofa", "description": "A sleek modern three-seater velvet sofa with tapered gold legs, ideal for contemporary living rooms.", "category": "sofa", "style": "modern", "room_type": "living room"},
    {"id": "prod_002", "name": "Scandinavian Fabric Couch", "description": "Minimalist Scandinavian-style couch in light gray fabric with wooden legs, perfect for clean open spaces.", "category": "sofa", "style": "scandinavian", "room_type": "living room"},
    {"id": "prod_003", "name": "Bohemian Floor Rug", "description": "Hand-woven bohemian area rug with intricate patterns in earthy tones, adds warmth and texture.", "category": "rug", "style": "bohemian", "room_type": "living room"},
    {"id": "prod_004", "name": "Industrial Arc Floor Lamp", "description": "Matte black industrial arc floor lamp with adjustable arm, perfect for reading corners.", "category": "lighting", "style": "industrial", "room_type": "living room"},
    {"id": "prod_005", "name": "Mid-Century Coffee Table", "description": "Walnut mid-century modern coffee table with hairpin legs and magazine shelf underneath.", "category": "table", "style": "modern", "room_type": "living room"},
    {"id": "prod_006", "name": "Smart 55\" OLED TV", "description": "Ultra-thin 55-inch OLED smart TV with built-in streaming apps and voice control.", "category": "electronics", "style": "modern", "room_type": "living room"},
    {"id": "prod_007", "name": "Floating Wall Shelf Set", "description": "Set of three floating wall shelves in natural oak, great for displaying decor and books.", "category": "shelving", "style": "scandinavian", "room_type": "living room"},
    {"id": "prod_008", "name": "Accent Armchair", "description": "Tufted accent armchair in emerald green velvet with brass nailhead trim.", "category": "chair", "style": "modern", "room_type": "living room"},

    # Bedroom
    {"id": "prod_009", "name": "Platform Bed Frame", "description": "Low-profile platform bed frame in solid walnut with integrated headboard and under-bed storage.", "category": "bed", "style": "modern", "room_type": "bedroom"},
    {"id": "prod_010", "name": "Minimalist Nightstand", "description": "Two-drawer minimalist nightstand in white lacquer with soft-close drawers.", "category": "table", "style": "minimalist", "room_type": "bedroom"},
    {"id": "prod_011", "name": "Linen Blackout Curtains", "description": "Heavyweight linen blend blackout curtains in charcoal gray for a restful sleep environment.", "category": "textile", "style": "minimalist", "room_type": "bedroom"},
    {"id": "prod_012", "name": "Bohemian Macramé Wall Hanging", "description": "Large hand-knotted macramé wall hanging in natural cotton, adds boho charm to any bedroom.", "category": "decor", "style": "bohemian", "room_type": "bedroom"},
    {"id": "prod_013", "name": "Scandinavian Dresser", "description": "Six-drawer oak dresser with rounded edges and brass knobs in Scandinavian design.", "category": "storage", "style": "scandinavian", "room_type": "bedroom"},

    # Dining Room
    {"id": "prod_014", "name": "Contemporary Dining Table", "description": "Extendable contemporary dining table in white marble top with brushed steel base, seats 6-8.", "category": "table", "style": "modern", "room_type": "dining room"},
    {"id": "prod_015", "name": "Upholstered Dining Chair Set", "description": "Set of four upholstered dining chairs in cream boucle fabric with oak legs.", "category": "chair", "style": "modern", "room_type": "dining room"},
    {"id": "prod_016", "name": "Statement Pendant Light", "description": "Oversized brass globe pendant light that serves as a dining room centerpiece.", "category": "lighting", "style": "modern", "room_type": "dining room"},
    {"id": "prod_017", "name": "Industrial Wine Rack", "description": "Wall-mounted industrial wine rack in wrought iron, holds 12 bottles with glass holder.", "category": "storage", "style": "industrial", "room_type": "dining room"},

    # Kitchen
    {"id": "prod_018", "name": "Bar Stool Set", "description": "Set of three adjustable-height swivel bar stools with faux leather seats and chrome base.", "category": "chair", "style": "modern", "room_type": "kitchen"},
    {"id": "prod_019", "name": "Open Kitchen Shelving Unit", "description": "Industrial-style open kitchen shelving in reclaimed wood and black iron pipe.", "category": "shelving", "style": "industrial", "room_type": "kitchen"},
    {"id": "prod_020", "name": "Ceramic Herb Planter Set", "description": "Set of four ceramic herb planters in matte white, perfect for kitchen windowsills.", "category": "decor", "style": "minimalist", "room_type": "kitchen"},

    # Office
    {"id": "prod_021", "name": "Ergonomic Office Chair", "description": "High-back ergonomic mesh office chair with lumbar support, adjustable armrests, and tilt mechanism.", "category": "chair", "style": "modern", "room_type": "office"},
    {"id": "prod_022", "name": "Standing Desk", "description": "Electric sit-stand desk with bamboo top and programmable height presets.", "category": "table", "style": "modern", "room_type": "office"},
    {"id": "prod_023", "name": "Desk Lamp with Wireless Charger", "description": "LED desk lamp with built-in wireless charging pad and adjustable color temperature.", "category": "lighting", "style": "modern", "room_type": "office"},
    {"id": "prod_024", "name": "Bookshelf Unit", "description": "Five-tier ladder bookshelf in natural bamboo with leaning design.", "category": "shelving", "style": "scandinavian", "room_type": "office"},

    # Bathroom
    {"id": "prod_025", "name": "Bamboo Bath Mat", "description": "Sustainable slatted bamboo bath mat with non-slip rubber grips.", "category": "textile", "style": "minimalist", "room_type": "bathroom"},
    {"id": "prod_026", "name": "Frameless LED Mirror", "description": "Wall-mounted frameless LED bathroom mirror with anti-fog, touch dimmer, and daylight simulation.", "category": "decor", "style": "modern", "room_type": "bathroom"},
    {"id": "prod_027", "name": "Floating Vanity Unit", "description": "Wall-mounted floating bathroom vanity in matte black with integrated basin and soft-close drawer.", "category": "storage", "style": "modern", "room_type": "bathroom"},

    # General / Multi-room
    {"id": "prod_028", "name": "Smart LED Strip Lights", "description": "WiFi-enabled RGB LED strip lights with app control and music sync, 16 million colors.", "category": "lighting", "style": "modern", "room_type": "any"},
    {"id": "prod_029", "name": "Ceramic Decorative Vase Set", "description": "Set of three handmade ceramic vases in earth tones, suitable for dried or fresh flower arrangements.", "category": "decor", "style": "bohemian", "room_type": "any"},
    {"id": "prod_030", "name": "Indoor Fiddle Leaf Fig Plant", "description": "Artificial 5-foot fiddle leaf fig tree in woven basket planter, maintenance-free greenery.", "category": "decor", "style": "modern", "room_type": "any"},
    {"id": "prod_031", "name": "Modular Storage Cubes", "description": "Stackable modular storage cubes in matte white with optional fabric drawer inserts.", "category": "storage", "style": "minimalist", "room_type": "any"},
    {"id": "prod_032", "name": "Vintage Wall Clock", "description": "Large vintage-style wall clock with Roman numerals and distressed bronze finish.", "category": "decor", "style": "industrial", "room_type": "any"},
    {"id": "prod_033", "name": "Eclectic Accent Table", "description": "Round accent side table with terrazzo top and geometric brass base.", "category": "table", "style": "eclectic", "room_type": "any"},
    {"id": "prod_034", "name": "Velvet Throw Pillow Set", "description": "Set of four velvet throw pillows in jewel tones — emerald, sapphire, amethyst, and ruby.", "category": "textile", "style": "modern", "room_type": "any"},
    {"id": "prod_035", "name": "Rattan Pendant Shade", "description": "Hand-woven rattan pendant lampshade that casts beautiful patterned shadows.", "category": "lighting", "style": "bohemian", "room_type": "any"},
]


def main():
    print(f"Seeding ChromaDB at {CHROMA_PATH} …")

    # Initialize
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare data
    ids = [item["id"] for item in FURNITURE_DATA]
    documents = [item["description"] for item in FURNITURE_DATA]
    metadatas = [
        {
            "name": item["name"],
            "category": item["category"],
            "style": item["style"],
            "room_type": item["room_type"],
        }
        for item in FURNITURE_DATA
    ]

    # Embed & upsert
    print(f"Embedding {len(documents)} descriptions …")
    embeddings = embedder.encode(documents).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"✅ Seeded {collection.count()} products into '{COLLECTION_NAME}' collection.")


if __name__ == "__main__":
    main()
