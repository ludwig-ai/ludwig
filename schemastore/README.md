# SchemaStore Submission Materials

This directory contains materials for submitting Ludwig's JSON Schema to
the [JSON Schema Store](https://www.schemastore.org/json/).

## Catalog Entry

The file `catalog-entry.json` contains the entry to add to SchemaStore's
`src/api/json/catalog.json`.

## Test Configs

The `test/` directory contains example Ludwig config files that validate
against the schema. These are used as positive test cases in the SchemaStore PR.

## How to Submit

1. Fork [SchemaStore/schemastore](https://github.com/SchemaStore/schemastore)
1. Add the catalog entry from `catalog-entry.json` to `src/api/json/catalog.json`
1. Copy test configs from `test/` to `src/test/ludwig/`
1. Submit a PR referencing Ludwig issue #1343
