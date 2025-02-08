## How to translate SPARQL into SQL?

Assumptions: 
- Nothing is NULL,
- Everything goes into the same SQL query.
- Predicate is always concrete.
- No transitive paths
- all filters translate to SQL

Each SPARQL var becomes either a row, or a value.
Each new subject introduces a row.
Each new object introduces a row if it's a join.
Each triple with a literal value adds a "where".
If the value is a variable:
  - Datatype property:
    - Variable selected?
      - Selection already bound?
        - The rows for the new bind and the old are the same?
          Add a where.
        - Different rows already joined:
          Add a where.
        - Different rows not yet joined:
          Add a join.
      - Selection not bound:
        - Bind selection
  - Object property:
    Treat as a datatype property on the pkey props. 
Each triple with a selectable variable value (object) binds the corresponding property expression to the value, if not already bound. 
Otherwise, adds a join if not already joined and if different rows.
Otherwise, adds to where.
Each filter contributes to "where".
If the value is a row, we have a join!

So, the algorithm:

- Consider a triple. We have a map of vars to expressions and a map of subjects to rows.
  - Identify the triple map(s) applicable to the triple:
    Subject, predicate, and object of the TM needs to fit with the triple.
  - Fork per map.
  - If the subject is new, introduce a new row.
  - If the object is new and a join, introduce a new row.
  - Generate the conditions.
  

### Simplest case: a single triple.

1. `select ?val { <subj> <prop> ?val }` where <prop> is a datatype property:

subj is a well-known template-based URI, identifyable by a primary key value <pkey_val>. prop is one of its fields. The table looks like:

```sql
create table T1 (
    pkey int primary key,
    prop varchar(50)
);
```

Result: `select t1.prop as val from T1 t1 where t1.pkey = <pkey_val>;`

How do we get there?
- Introduce a new row for <subj>. 
- Generate `t1.pkey = <pkey_val>` from <subj>
- Generate `select t1.prop as val` from <prop> and ?val.

2. `select ?subj { ?subj <prop> <val> }`

Result: `select build_uri(template_T1, t1.pkey) as subj from T1 t1 where t1.prop = <val>;`

3. `select ?subj { ?subj a T1 }`

Result: `select build_uri(template_T1, subj.pkey) from T1 subj;`

4. `select ?vobj { <subj> <prop> ?vobj } where <prop> is an object property. The tables look like this:

```sql
create table T1 (
    pkey int primary key,
    prop int,
    constraint foreign key prop references T2(pkey2)
)

create table T2 (
    pkey2 int primary key
)

Result: `select build_uri(template_T2, t2.pkey2) from T1 t1 join T2 t2 on t1.prop = t2.pkey2 where t1.pkey = <pkey_val>;`

5. `select ?subj { ?subj <prop> <vobj> }` where <prop> is an object property. 

Result: `select build_uri(template_T1, t1.pkey) from T1 t1 join T2 t2 on t1.prop = t2.pkey where t2.pkey2 = <pkey2_val>;`

### BGP: multiple triples

- Grow one by one.
- If a variable is free, see above.
- Otherwise, see above (if the subject/value is a variable.)
