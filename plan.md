### R2RML mapping.

- [v] Mapping for rdf:type.
- [v] Support for joins (explicit via parentTriplesMap, e.g. shippers_of_order).
- [v] Support for properties with multiple domains (e.g., Demo:city)
- [v] Fix unions
- [v] Support for blank nodes
- [v] Support for sequence, alt, inv paths 
- [ ] Unpack joins into BGPs
- [ ] Support for better equality of template-based objects (concats)

### Some day
- [ ] Support for retrieving types (triples like `(<something> a ?var)`)
- [ ] Support for joins (implicit via matching templates, e.g. orders_has_customers) - this is avoidable by correct R2RML construction.
- [ ] Support for OPTIONAL
- [ ] Support for FILTER NOT EXISTS
- [ ] Support for opt paths
- [ ] Check that the IN op works when the set elements are not literals