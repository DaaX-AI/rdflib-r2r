### R2RML mapping.

- [v] Mapping for rdf:type.
- [v] Support for joins (explicit via parentTriplesMap, e.g. shippers_of_order).
- [ ] Support for properties with multiple domains (e.g., Demo:city)
- [ ] Support for paths.

### Some day
- [ ] Support for retrieving types (triples like `(<something> a ?var)`)
- [ ] Support for joins (implicit via matching templates, e.g. orders_has_customers) - this is avoidable by correct R2RML construction.