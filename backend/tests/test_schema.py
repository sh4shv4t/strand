from app.schema import Garment, ImageRecord, ParsedQuery, QueryRequest


def test_garment_color_is_optional():
    garment = Garment(slot="upper", type="shirt")
    assert garment.color is None


def test_image_record_defaults():
    record = ImageRecord(id="x1", garments=[], caption="a caption")
    assert record.scene is None
    assert record.style is None
    assert record.notable == []
    assert record.swatch == []


def test_parsed_query_defaults():
    parsed = ParsedQuery(raw_query="a query")
    assert parsed.garments == []
    assert parsed.scene is None
    assert parsed.confidence == 1.0


def test_query_request_defaults():
    request = QueryRequest(query="a query")
    assert request.top_k == 5
    assert request.alpha == 0.6
