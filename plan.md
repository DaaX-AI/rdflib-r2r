### R2RML mapping.

### Bugs
- [ ] dates come out as strings (but maybe that's OK if we convert back to SQL?)
- [v] In a SPARQL join, class info should propagate between branches.

### Features and improvements
- [v] Left joins
- [v] Exists
- [v] Unpack joins into BGPs
- [v] Support for better equality of template-based objects (concats)
- [ ] Use property range and domain information to determine classes
- [ ] Do not create extra copies of a table when a main and an auxiliary triple maps are used.

### Some day
- [ ] Support for retrieving types (triples like `(<something> a ?var)`)
- [ ] Support for joins (implicit via matching templates, e.g. orders_has_customers) - this is avoidable by correct R2RML construction.
- [v] Support for OPTIONAL
- [v] Support for FILTER NOT EXISTS
- [ ] Support for opt SPARQL paths
- [ ] Check that the IN op works when the set elements are not literals
- [ ] Support for transitive SPARQL paths via recursive CTEs