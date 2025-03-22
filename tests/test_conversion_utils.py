import pytest
from sqlalchemy import Column, ColumnElement, Integer, String, create_engine, MetaData, Table
from sqlalchemy.sql import literal
from rdflib_r2r.conversion_utils import try_match_templates, format_template, TemplateInfo

@pytest.fixture
def engine():
    return create_engine('sqlite:///:memory:')

@pytest.fixture
def metadata():
    return MetaData()

@pytest.fixture
def test_table(engine, metadata):
    table = Table('test', metadata,
        Column('id', Integer, primary_key=True),
        Column('name', String),
        Column('age', Integer)
    )
    metadata.create_all(engine)
    return table

def ce2str(ce: ColumnElement) -> str:
    return str(ce.compile(compile_kwargs={"literal_binds": True}))

def test_template_to_literal_match(test_table):
    # Create a template column
    template_col = format_template("http://example.org/person/{id}", test_table, True)
    
    # Test matching literal
    literal_col = literal("http://example.org/person/123")
    result = try_match_templates(template_col, literal_col, True)
    assert result is not None
    assert len(result) == 1
    assert ce2str(result[0]) == "test.id = '123'"

def test_template_to_literal_mismatch(test_table):
    # Create a template column
    template_col = format_template("http://example.org/person/{id}", test_table, True)
    
    # Test non-matching literal
    literal_col = literal("http://example.org/other/123")
    with pytest.raises(Exception) as exc_info:
        try_match_templates(template_col, literal_col, True)
    assert "Template mismatch" in str(exc_info.value)

def test_template_to_literal_inequality(test_table):
    # Create a template column
    template_col = format_template("http://example.org/person/{id}", test_table, True)
    
    # Test inequality with non-matching literal
    literal_col = literal("http://example.org/other/123")
    result = try_match_templates(template_col, literal_col, False)
    assert result == []

def test_template_to_template_match(test_table):
    # Create two template columns with same template
    template1 = format_template("http://example.org/person/{id}", test_table, True)
    template2 = format_template("http://example.org/person/{id}", test_table, True)
    
    result = try_match_templates(template1, template2, True)
    assert result is not None
    assert len(result) == 1
    assert str(result[0]) == "test.id = test.id"

def test_template_to_template_mismatch(test_table):
    # Create two template columns with different templates
    template1 = format_template("http://example.org/person/{id}", test_table, True)
    template2 = format_template("http://example.org/other/{id}", test_table, True)
    
    with pytest.raises(Exception) as exc_info:
        try_match_templates(template1, template2, True)
    assert "Template mismatch" in str(exc_info.value)

def test_template_to_template_inequality(test_table):
    # Create two template columns with different templates
    template1 = format_template("http://example.org/person/{id}", test_table, True)
    template2 = format_template("http://example.org/other/{id}", test_table, True)
    
    result = try_match_templates(template1, template2, False)
    assert result == []

def test_non_template_columns(test_table):
    # Test with regular columns
    col1 = test_table.c.id
    col2 = test_table.c.id
    
    result = try_match_templates(col1, col2, True)
    assert result is None

def test_template_with_multiple_columns(test_table):
    # Create a template with multiple columns
    template_col = format_template("http://example.org/person/{id}/{name}", test_table, True)
    
    # Test matching literal
    literal_col = literal("http://example.org/person/123/John")
    result = try_match_templates(template_col, literal_col, True)
    assert result is not None
    assert len(result) == 2
    assert ce2str(result[0]) == "test.id = '123'"
    assert ce2str(result[1]) == "test.name = 'John'"