import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = process.cwd();
const taxDir = path.join(projectRoot, "data", "tax");
const dataPath = path.join(taxDir, "tax_2025_workbook_data.json");
const outputPath = path.join(taxDir, "tax_2025_capital_gains_all_accounts.xlsx");

const raw = JSON.parse(await fs.readFile(dataPath, "utf8"));

function rowsToMatrix(rows) {
  if (!rows || rows.length === 0) return [["No rows"]];
  const headers = Object.keys(rows[0]);
  return [headers, ...rows.map((row) => headers.map((header) => row[header] ?? ""))];
}

function writeMatrix(sheet, startCell, matrix) {
  if (startCell !== "A1") {
    throw new Error("writeMatrix currently expects A1 start.");
  }
  const range = sheet.getRangeByIndexes(0, 0, matrix.length, matrix[0].length);
  range.values = matrix;
  return range;
}

function formatTable(sheet, matrix, options = {}) {
  const range = sheet.getRangeByIndexes(0, 0, matrix.length, matrix[0].length);
  const header = sheet.getRangeByIndexes(0, 0, 1, matrix[0].length);
  header.format.fill.color = options.headerColor ?? "#1f4e78";
  header.format.font.color = "#ffffff";
  header.format.font.bold = true;
  range.format.font.name = "Aptos";
  range.format.font.size = 10;
  sheet.freezePanes.freezeRows(1);
  sheet.getUsedRange().format.autofitColumns();
}

function addSheet(workbook, name, rows, options = {}) {
  const sheet = workbook.worksheets.add(name);
  const matrix = rowsToMatrix(rows);
  writeMatrix(sheet, "A1", matrix);
  formatTable(sheet, matrix, options);
  return sheet;
}

function summaryRows() {
  const methods = raw.method_summary.filter((row) => row.term === "Covered total");
  const recommendedMethod = "FIFO_PRIOR_HIFO_2025";
  const recommendedRows = raw.method_summary.filter((row) => row.method === recommendedMethod);
  const shortGain = raw.summary.short_closures.reduce((sum, row) => sum + Number(row.realized_gain || 0), 0);
  return [
    { item: "Tax year", value: raw.summary.tax_year, unit: "", notes: "" },
    { item: "Source file", value: raw.summary.source, unit: "", notes: "Futu history_deal_list_query export" },
    {
      item: "Recommended lot method scenario",
      value: recommendedMethod,
      unit: "",
      notes: "FIFO through prior years, then highest cost first for 2025 sales.",
    },
    {
      item: "FIFO covered realized gain, HKD",
      value: methods.find((r) => r.method === "FIFO" && r.currency === "HKD")?.realized_gain ?? 0,
      unit: "HKD",
      notes: "Stock trades only; excludes uncovered basis sales.",
    },
    {
      item: "Recommended covered realized gain, HKD",
      value: methods.find((r) => r.method === recommendedMethod && r.currency === "HKD")?.realized_gain ?? 0,
      unit: "HKD",
      notes: "Kept in HKD; no FX conversion applied.",
    },
    {
      item: "FIFO covered realized gain, USD",
      value: methods.find((r) => r.method === "FIFO" && r.currency === "USD")?.realized_gain ?? 0,
      unit: "USD",
      notes: "Stock trades only.",
    },
    {
      item: "Recommended covered realized gain, USD",
      value: methods.find((r) => r.method === recommendedMethod && r.currency === "USD")?.realized_gain ?? 0,
      unit: "USD",
      notes: "Stock trades only.",
    },
    {
      item: "Short option closure realized gain, USD",
      value: shortGain,
      unit: "USD",
      notes: "Two BUY_BACK closures at zero cost; one short option remains open after 2025.",
    },
    {
      item: "Uncovered proceeds, HKD",
      value: recommendedRows.find((r) => r.currency === "HKD" && r.term === "Unknown basis")?.uncovered_proceeds ?? 0,
      unit: "HKD",
      notes: "Needs pre-2024 cost basis.",
    },
    {
      item: "Uncovered proceeds, USD",
      value: recommendedRows.find((r) => r.currency === "USD" && r.term === "Unknown basis")?.uncovered_proceeds ?? 0,
      unit: "USD",
      notes: "Needs pre-2024 cost basis.",
    },
    {
      item: "Wash sale candidate rows",
      value: raw.summary.wash_sale_candidate_count,
      unit: "rows",
      notes: "Candidate screen only; losses may be deferred.",
    },
  ];
}

function assumptionsRows() {
  return [
    {
      topic: "Data source",
      assumption: "Only Futu historical deal records are used.",
      implication: "Current holdings and watchlists are not used.",
    },
    {
      topic: "Lot methods",
      assumption: "FIFO and HIFO are simulated from known lots available before each sale.",
      implication: "HIFO approximates tax-minimizing specific identification.",
    },
    {
      topic: "Specific identification",
      assumption: "Specific lots require broker instruction/confirmation at sale time.",
      implication: "If unavailable, FIFO may be the defensible default.",
    },
    {
      topic: "Currency",
      assumption: "HKD and USD are not combined in this workbook.",
      implication: "No FX conversion is applied; handle final conversion separately.",
    },
    {
      topic: "Fees",
      assumption: "Futu deal history export did not include commissions, platform fees, SEC fees, ADR fees, or stamp duty.",
      implication: "Final taxable gain should adjust proceeds/basis for reportable fees.",
    },
    {
      topic: "Uncovered basis",
      assumption: "Some 2025 sales exceed buys found in the 2024-2025 export.",
      implication: "Pre-2024 purchase history or broker 1099 basis is required.",
    },
    {
      topic: "Wash sales",
      assumption: "Candidate screen flags same-symbol buys within +/-30 days of HIFO loss allocations.",
      implication: "Final adjustment needs all accounts and substantially-identical security review.",
    },
    {
      topic: "Tax amount",
      assumption: "Workbook calculates realized gains, not final tax due.",
      implication: "Tax due depends on filing status, ordinary income, carryovers, NIIT, and wash-sale/basis/FX adjustments.",
    },
  ];
}

const workbook = Workbook.create();

addSheet(workbook, "Summary", summaryRows(), { headerColor: "#17365d" });
addSheet(workbook, "Assumptions", assumptionsRows(), { headerColor: "#806000" });
addSheet(workbook, "Method Summary", raw.method_summary);
addSheet(workbook, "Symbol Summary", raw.symbol_summary);
addSheet(workbook, "Uncovered Basis", raw.uncovered_basis, { headerColor: "#7f1d1d" });
addSheet(workbook, "Wash Sale Candidates", raw.wash_sale_candidates, { headerColor: "#7f1d1d" });
addSheet(workbook, "Short Closures", raw.short_closures);
addSheet(workbook, "Lot Allocations", raw.lot_allocations);

for (const sheet of workbook.worksheets) {
  sheet.showGridLines = false;
}

await fs.mkdir(taxDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

console.log(outputPath);
