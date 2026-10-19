"""Company annual reports from investor-relations sites, listed explicitly (controller ruling, in
place of EDGAR Form ARS, which www.sec.gov refuses to the project User-Agent).

Every URL below is a company's own published annual report (a glossy annual report, an annual
report and accounts, or a 10-K or 20-F the company hosts) for fiscal years 2019-2025, with the
business overview, management discussion and financial statements. They were found on 2026-10-19 by
reading companies' IR "annual reports" pages (robots.txt obeyed, 2 s between requests to a host),
a few web searches and the same file-name pattern for other years, then kept only if a
one-kilobyte range request with the project User-Agent answered with a ``%PDF-`` file of at
most 50 MB. Half-year reports, summaries, sustainability and non-financial reports, slide decks,
regulatory capital disclosures and duplicate renderings (interactive, spreads, accessible) were
left out; at most two reports per company and year.

The documents belong to their publishers: they are kept locally for research and **never
redistributed**; the manifest holds only URLs and hashes. Some dev or held-out documents are
annual reports, so the fetcher's exclusion by sha256, URL and title applies here as everywhere.
Candidates are interleaved by company, so any quota spans many companies.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterator

from benchmarks.training_corpus._net import Net
from benchmarks.training_corpus.sources.base import Candidate, Source, round_robin, safe_id

NAME = "ir_annual_reports"
LICENCE = "company annual report from its IR site; local research use only, never redistributed"

# (publisher, fiscal year, URL)
REPORTS: tuple[tuple[str, int, str], ...] = (
    (
        "ANZ Group",
        2025,
        "https://www.anz.com/content/dam/anzcom/shareholder/2025-annual-report/anz-2025-annual-report.pdf",
    ),
    (
        "ANZ Group",
        2025,
        "https://www.anz.com/content/dam/anzcom/shareholder/2025-annual-report/anzbgl-2025-annual-report.pdf",
    ),
    (
        "AT&T",
        2025,
        "https://investors.att.com/~/media/Files/A/ATT-IR-V2/financial-reports/annual-reports/2025/2025-annual-report-complete.pdf",
    ),
    (
        "Atlas Copco",
        2020,
        "https://www.atlascopcogroup.com/content/dam/atlas-copco/group/documents/investors/financial-publications/english/20210311-annual-report-2020.pdf",
    ),
    (
        "Atlas Copco",
        2021,
        "https://www.atlascopcogroup.com/content/dam/atlas-copco/group/documents/investors/financial-publications/english/20220321-annual-report-2021.pdf",
    ),
    (
        "Aviva",
        2025,
        "https://static.aviva.io/content/dam/aviva-corporate/documents/investors/pdfs/reports/2025/annual-report-and-accounts-2025.pdf",
    ),
    (
        "Barclays",
        2019,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/reports-and-events/annual-reports/2018/2018-Barclays-Bank-UK-PLC-Annual-Report-28.02.2019.pdf",
    ),
    (
        "Barclays",
        2019,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/reports-and-events/annual-reports/2019/Barclays%20Bank%20Ireland%20PLC%20Annual%20Report%202019.pdf",
    ),
    (
        "Barclays",
        2020,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/reports-and-events/annual-reports/2020/BB-PLC-2020-AR.pdf",
    ),
    (
        "Barclays",
        2020,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/reports-and-events/annual-reports/2020/Barclays%20Bank%20Ireland%20PLC%20Annual%20Report%202020.pdf",
    ),
    (
        "Barclays",
        2021,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/reports-and-events/annual-reports/2021/BBI-PLC-AR-2021.pdf",
    ),
    (
        "Barclays",
        2021,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/reports-and-events/annual-reports/2021/Barclays-Bank-PLC-2021-AR.pdf",
    ),
    (
        "Barclays",
        2022,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/reports-and-events/annual-reports/2022/AR/BBI-PLC-AR-2022.pdf",
    ),
    (
        "Barclays",
        2022,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/reports-and-events/annual-reports/2022/AR/Barclays-Bank-PLC-AR-2022.pdf",
    ),
    (
        "Barclays",
        2023,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/reports-and-events/annual-reports/2023/Barclays%20Bank%20Ireland%20PLC%20Annual%20Report%202023.pdf",
    ),
    (
        "Barclays",
        2023,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/reports-and-events/annual-reports/2023/Barclays-Bank-PLC-Annual-Report-2023.pdf",
    ),
    (
        "Barclays",
        2024,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/ResultAnnouncements/FullYear2024Results/BCSL_Annual_Report_2024.pdf",
    ),
    (
        "Barclays",
        2024,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/reports-and-events/annual-reports/2024/Barclays%20Bank%20Ireland%20PLC%20Annual%20Report%202024.pdf",
    ),
    (
        "Barclays",
        2025,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/ResultAnnouncements/FullYear2025Results/BCSL-Annual-Report-2025.pdf",
    ),
    (
        "Barclays",
        2025,
        "https://home.barclays/content/dam/home-barclays/documents/investor-relations/reports-and-events/annual-reports/2025/Barclays-Bank-PLC-Annual-Report-2025.pdf",
    ),
    ("Berkshire Hathaway", 2019, "https://www.berkshirehathaway.com/2019ar/2019ar.pdf"),
    ("Berkshire Hathaway", 2020, "https://www.berkshirehathaway.com/2020ar/2020ar.pdf"),
    ("Berkshire Hathaway", 2021, "https://www.berkshirehathaway.com/2021ar/2021ar.pdf"),
    ("Berkshire Hathaway", 2022, "https://www.berkshirehathaway.com/2022ar/2022ar.pdf"),
    ("Berkshire Hathaway", 2023, "https://www.berkshirehathaway.com/2023ar/2023ar.pdf"),
    ("Berkshire Hathaway", 2024, "https://www.berkshirehathaway.com/2024ar/2024ar.pdf"),
    ("Berkshire Hathaway", 2025, "https://www.berkshirehathaway.com/2025ar/2025ar.pdf"),
    (
        "Bristol Myers Squibb",
        2019,
        "https://www.bms.com/assets/bms-ar/documents/2019/2019-10-K.pdf",
    ),
    (
        "Bristol Myers Squibb",
        2019,
        "https://www.bms.com/assets/bms-ar/documents/2019/2019-BMS-Annual-Report.pdf",
    ),
    (
        "Bristol Myers Squibb",
        2020,
        "https://www.bms.com/assets/bms-ar/documents/2020/BMS_10k_new.pdf",
    ),
    (
        "Bristol Myers Squibb",
        2020,
        "https://www.bms.com/assets/bms-ar/documents/2020/Final_BMS_2020_Annual_Report.pdf",
    ),
    (
        "Bristol Myers Squibb",
        2021,
        "https://annual-report.bms.com/assets/bms-ar/documents/2021-annual-report.pdf",
    ),
    (
        "Bristol Myers Squibb",
        2021,
        "https://annual-report.bms.com/assets/bms-ar/documents/bms-2021-10-K.pdf",
    ),
    (
        "Bristol Myers Squibb",
        2022,
        "https://annual-report.bms.com/assets/bms-ar/documents/2022-annual-report.pdf",
    ),
    (
        "Bristol Myers Squibb",
        2022,
        "https://annual-report.bms.com/assets/bms-ar/documents/2022-bms-financial-report.pdf",
    ),
    ("Bristol Myers Squibb", 2022, "https://www.bms.com/assets/bms-ar/documents/bms-2022-10-K.pdf"),
    (
        "Bristol Myers Squibb",
        2023,
        "https://www.bms.com/assets/bms-ar/documents/2023/2023-bms-financial-report.pdf",
    ),
    ("Bristol Myers Squibb", 2023, "https://www.bms.com/assets/bms-ar/documents/bms-2023-10-K.pdf"),
    (
        "Bristol Myers Squibb",
        2024,
        "https://www.bms.com/assets/bms-ar/documents/2024/2024-bms-annual-report.pdf",
    ),
    (
        "Bristol Myers Squibb",
        2024,
        "https://www.bms.com/assets/bms-ar/documents/2024/bms-2024-10-K.pdf",
    ),
    (
        "Bristol Myers Squibb",
        2025,
        "https://www.bms.com/assets/bms-ar/documents/2025/bms-2025-10-K.pdf",
    ),
    (
        "Brookfield Asset Management",
        2022,
        "https://bam.brookfield.com/sites/brookfield-bam-v2/files/brookfield-bam-v2/report-fillings/annual-quarterly-reports/2022-bam-annual-report-v2.pdf",
    ),
    (
        "Brookfield Asset Management",
        2023,
        "https://bam.brookfield.com/sites/brookfield-bam-v2/files/brookfield-bam-v2/report-fillings/annual-quarterly-reports/bam-2023-annual-report-vf.pdf",
    ),
    ("Bunzl", 2019, "https://www.bunzl.com/media/sgkpn1rq/bunzl_ar_2019.pdf"),
    ("Bunzl", 2020, "https://www.bunzl.com/media/qsicxgbr/bunzl_ar_2020.pdf"),
    ("Bunzl", 2023, "https://www.bunzl.com/media/0aljxph1/annual-report-2023.pdf"),
    (
        "CIBC",
        2019,
        "https://www.cibc.com/content/dam/about_cibc/investor_relations/pdfs/quarterly_results/2019/ar-19-en.pdf",
    ),
    (
        "CIBC",
        2020,
        "https://www.cibc.com/content/dam/about_cibc/investor_relations/pdfs/quarterly_results/2020/ar-20-en.pdf",
    ),
    (
        "CIBC",
        2021,
        "https://www.cibc.com/content/dam/about_cibc/investor_relations/pdfs/quarterly_results/2021/ar-21-en.pdf",
    ),
    (
        "CIBC",
        2022,
        "https://www.cibc.com/content/dam/cibc-public-assets/about-cibc/investor-relations/pdfs/quarterly-results/2022/ar-22-en.pdf",
    ),
    (
        "CIBC",
        2023,
        "https://www.cibc.com/content/dam/cibc-public-assets/about-cibc/investor-relations/pdfs/quarterly-results/2023/ar-23-en.pdf",
    ),
    (
        "CIBC",
        2024,
        "https://www.cibc.com/content/dam/cibc-public-assets/about-cibc/investor-relations/pdfs/quarterly-results/2024/ar-24-en.pdf",
    ),
    (
        "CIBC",
        2025,
        "https://www.cibc.com/content/dam/cibc-public-assets/about-cibc/investor-relations/pdfs/quarterly-results/2025/ar-25-en.pdf",
    ),
    (
        "CRH",
        2019,
        "https://www.crh.com/wp-content/uploads/2025/06/crh-annual-report-2019-1-1-1-2-1-1-1-1-1-1.pdf",
    ),
    (
        "CRH",
        2021,
        "https://www.crh.com/wp-content/uploads/2025/06/crh-annual-report-2021-1-1-1-1-2-1-1-1-1-1-1.pdf",
    ),
    (
        "CRH",
        2023,
        "https://www.crh.com/wp-content/uploads/2025/06/crh-annual-report-2023-web-version-1-1-1-1-2-1-1-1-1-1-1.pdf",
    ),
    (
        "CRH",
        2024,
        "https://www.crh.com/wp-content/uploads/2025/06/crh-2024-annual-report-1-1-1-1-2-1-1-1-1-1-1.pdf",
    ),
    ("CRH", 2025, "https://www.crh.com/wp-content/uploads/2026/03/2025-Annual-Report.pdf"),
    (
        "Citigroup",
        2022,
        "https://www.citigroup.com/rcs/citigpa/storage/public/citi-2022-annual-report.pdf",
    ),
    (
        "Citigroup",
        2023,
        "https://www.citigroup.com/rcs/citigpa/storage/public/Citi-2023-Annual-Report.pdf",
    ),
    (
        "Citigroup",
        2024,
        "https://www.citigroup.com/rcs/citigpa/storage/public/citi-2024-annual-report.pdf",
    ),
    (
        "Citigroup",
        2025,
        "https://www.citigroup.com/rcs/citigpa/storage/public/Annual_Report/2025/citi-2025-annual-report.pdf",
    ),
    (
        "Conduit Holdings",
        2025,
        "https://www.conduitreinsurance.com/media/kwgni3yo/conduit-holdings-annual-report-accounts-2025.pdf",
    ),
    (
        "Croda International",
        2025,
        "https://www.croda.com/mediaassets/files/corporate/annual-report-2025/croda-annual-report-2025.pdf?la=en-GB&hash=49A5CFC09C17CE953B5D58AF6F8F6F24",
    ),
    (
        "Deutsche Bank",
        2020,
        "https://investor-relations.db.com/files/documents/annual-reports/Annual_Report_2020.pdf?language_id=1",
    ),
    (
        "Deutsche Bank",
        2021,
        "https://investor-relations.db.com/files/documents/annual-reports/2022/Annual_Report_2021.pdf?language_id=1",
    ),
    (
        "Deutsche Bank",
        2022,
        "https://investor-relations.db.com/files/documents/annual-reports/2023/Annual-Report-2022.pdf",
    ),
    (
        "Deutsche Bank",
        2023,
        "https://investor-relations.db.com/files/documents/annual-reports/2024/Annual-Report-2023.pdf",
    ),
    (
        "Deutsche Bank",
        2024,
        "https://investor-relations.db.com/files/documents/other-presentations-and-events/2024/Annual-Report-2024.pdf",
    ),
    (
        "Deutsche Bank",
        2025,
        "https://investor-relations.db.com/files/documents/other-presentations-and-events/2025/Annual-Report-2025.pdf?language_id=1",
    ),
    ("Diploma", 2019, "https://www.diplomaplc.com/media/untlk05o/2019-annual-report-accounts.pdf"),
    (
        "Diploma",
        2022,
        "https://www.diplomaplc.com/media/211pjsdn/diploma-plc-annual-report-accounts-2022.pdf",
    ),
    ("Diploma", 2023, "https://www.diplomaplc.com/media/rkej2axs/diploma_2023-annual-report.pdf"),
    ("Diploma", 2024, "https://www.diplomaplc.com/media/vdyeuqza/diploma-2024-annual-report.pdf"),
    (
        "Diploma",
        2025,
        "https://www.diplomaplc.com/media/5s3fi0yk/diploma-plc-annual-report-and-accounts-2025.pdf",
    ),
    (
        "Erste Group",
        2019,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/2019/Reports/AR2019_Annual_Financial_Report_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2019,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/2019/Reports/AR2019_FINAL_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2020,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/2020/Reports/AR2020_Annual_Financial_Report_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2020,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/2020/Reports/AR2020_FINAL_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2021,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/2021/Reports/AR2021_Annual_Financial_Report_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2021,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/2021/Reports/AR2021_FINAL_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2022,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/2022/Reports/AR2022_Annual_Financial_Report_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2022,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/2022/Reports/AR2022_FINAL_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2023,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/onlineAR2023/ar23reports/AR2023_Annual_financial_report_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2023,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/onlineAR2023/ar23reports/AR2023_FINAL_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2024,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/onlinear2024/ar24reports/AR2024_Annual_financial_report_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2024,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/onlinear2024/ar24reports/AR2024_FINAL_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2025,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/onlinear2025/ar25reports/AR2025_Annual_financial_report_en.pdf?forceDownload=1",
    ),
    (
        "Erste Group",
        2025,
        "https://cdn.erstegroup.com/content/dam/at/eh/www_erstegroup_com/en/Investor_Relations/onlinear2025/ar25reports/AR2025_FINAL_en.pdf?forceDownload=1",
    ),
    ("Flutter Entertainment", 2024, "https://flutter.com/media/v1ijhmyw/flutter-2024-10-k.pdf"),
    (
        "Flutter Entertainment",
        2025,
        "https://flutter.com/media/le5b5mp4/2025-annual-report-and-accounts.pdf",
    ),
    ("GSK", 2025, "https://www.gsk.com/media/kn0bknmd/annual-report-2025.pdf"),
    ("General Electric", 2022, "https://www.ge.com/sites/default/files/ge_ar2022_annualreport.pdf"),
    ("General Electric", 2023, "https://www.ge.com/sites/default/files/ge_ar2023_annualreport.pdf"),
    (
        "Goldman Sachs",
        2021,
        "https://www.goldmansachs.com/investor-relations/financials/current/annual-reports/2021-annual-report/multimedia/annual-report-2021.pdf",
    ),
    (
        "Goldman Sachs",
        2022,
        "https://www.goldmansachs.com/investor-relations/financials/current/annual-reports/2022-annual-report/multimedia/annual-report-2022.pdf",
    ),
    (
        "HSBC Holdings",
        2024,
        "https://www.hsbc.com/-/files/hsbc/investors/hsbc-results/2024/annual/pdfs/hsbc-holdings-plc/250219-annual-report-and-accounts-2024.pdf",
    ),
    (
        "HSBC Holdings",
        2025,
        "https://www.hsbc.com/-/files/hsbc/investors/hsbc-results/2025/annual/pdfs/hsbc-holdings-plc/260225-annual-report-and-accounts-2025.pdf",
    ),
    (
        "Holcim",
        2019,
        "https://www.holcim.com/sites/holcim/files/documents/02272020-finance-lafageholcim_fy_2019_report-en_421281078.pdf",
    ),
    (
        "Holcim",
        2020,
        "https://www.holcim.com/sites/holcim/files/documents/26022021-finance-lafageholcim_fy_2020_report-full-en.pdf",
    ),
    (
        "Holcim",
        2021,
        "https://www.holcim.com/sites/holcim/files/2022-03/25022022-finance-holcim_fy_2021_report-full-en.pdf",
    ),
    (
        "Holcim",
        2022,
        "https://www.holcim.com/sites/holcim/files/2023-02/24022023-finance-holcim-fy-2022-report-full-en-3914999618.pdf",
    ),
    (
        "Holcim",
        2024,
        "https://www.holcim.com/sites/holcim/files/docs/28022025-finance-holcim-fy-2024-report-full-en.pdf",
    ),
    (
        "Holcim",
        2025,
        "https://www.holcim.com/sites/holcim/files/docs/27022026-finance-holcim-fy-2025-report-full-en.pdf",
    ),
    ("IBM", 2019, "https://www.ibm.com/investor/att/pdf/IBM_Annual_Report_2019.pdf"),
    ("IBM", 2020, "https://www.ibm.com/investor/att/pdf/IBM_Annual_Report_2020.pdf"),
    ("IBM", 2021, "https://www.ibm.com/investor/att/pdf/IBM_Annual_Report_2021.pdf"),
    ("IBM", 2022, "https://www.ibm.com/investor/att/pdf/IBM_Annual_Report_2022.pdf"),
    (
        "Informa",
        2019,
        "https://www.informa.com/globalassets/documents/investor-relations/2020/informa-annual-report-2019.pdf",
    ),
    (
        "Informa",
        2020,
        "https://www.informa.com/globalassets/documents/investor-relations/2021/informa-annual-report--financial-statements-2020.pdf",
    ),
    (
        "Informa",
        2021,
        "https://www.informa.com/globalassets/documents/investor-relations/2022/informa-annual-report-2021.pdf",
    ),
    (
        "Informa",
        2022,
        "https://www.informa.com/globalassets/documents/investor-relations/2023/informa-annual-report-2022.pdf",
    ),
    (
        "Informa",
        2023,
        "https://www.informa.com/globalassets/documents/investor-relations/2024/2023-informa-annual-report.pdf",
    ),
    (
        "Informa",
        2024,
        "https://www.informa.com/globalassets/documents/investor-relations/2025/2024-informa-annual-report.pdf",
    ),
    (
        "Informa",
        2025,
        "https://www.informa.com/globalassets/documents/investor-relations/2026/2025-informa-annual-report.pdf",
    ),
    (
        "Intel",
        2024,
        "https://www.intc.com/filings-reports/all-sec-filings/content/0000050863-25-000052/a2024arsform10-k.pdf",
    ),
    (
        "International Airlines Group",
        2025,
        "https://www.iairgroup.com/media/ktnlp1jx/iag-annual-report-and-accounts-2025.pdf",
    ),
    (
        "International Airlines Group",
        2025,
        "https://www.iairgroup.com/media/pwxlav5v/individual-annual-financial-report-2025.pdf",
    ),
    (
        "JPMorgan Chase",
        2019,
        "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2019.pdf",
    ),
    (
        "JPMorgan Chase",
        2020,
        "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2020.pdf",
    ),
    (
        "JPMorgan Chase",
        2021,
        "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2021.pdf",
    ),
    (
        "JPMorgan Chase",
        2022,
        "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2022.pdf",
    ),
    (
        "JPMorgan Chase",
        2023,
        "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2023.pdf",
    ),
    (
        "JPMorgan Chase",
        2024,
        "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2024.pdf",
    ),
    (
        "JPMorgan Chase",
        2025,
        "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2025.pdf",
    ),
    (
        "Lockheed Martin",
        2019,
        "https://www.lockheedmartin.com/content/dam/lockheed-martin/eo/documents/annual-reports/lockheed-martin-annual-report-2019.pdf",
    ),
    (
        "Lockheed Martin",
        2020,
        "https://www.lockheedmartin.com/content/dam/lockheed-martin/eo/documents/annual-reports/lockheed-martin-annual-report-2020.pdf",
    ),
    (
        "Lockheed Martin",
        2021,
        "https://www.lockheedmartin.com/content/dam/lockheed-martin/eo/documents/annual-reports/lockheed-martin-annual-report-2021.pdf",
    ),
    (
        "Lockheed Martin",
        2022,
        "https://www.lockheedmartin.com/content/dam/lockheed-martin/eo/documents/annual-reports/lockheed-martin-annual-report-2022.pdf",
    ),
    (
        "Lockheed Martin",
        2023,
        "https://www.lockheedmartin.com/content/dam/lockheed-martin/eo/documents/annual-reports/lockheed-martin-annual-report-2023.pdf",
    ),
    (
        "Lockheed Martin",
        2024,
        "https://www.lockheedmartin.com/content/dam/lockheed-martin/eo/documents/annual-reports/lockheed-martin-annual-report-2024.pdf",
    ),
    (
        "Lockheed Martin",
        2025,
        "https://www.lockheedmartin.com/content/dam/lockheed-martin/eo/documents/annual-reports/lockheed-martin-annual-report-2025.pdf",
    ),
    (
        "London Stock Exchange Group",
        2019,
        "https://www.lseg.com/content/dam/lseg/en_us/documents/investor-relations/annual-reports/lseg-annual-report-2019.pdf",
    ),
    (
        "London Stock Exchange Group",
        2020,
        "https://www.lseg.com/content/dam/lseg/en_us/documents/investor-relations/annual-reports/lseg-annual-report-2020.pdf",
    ),
    (
        "London Stock Exchange Group",
        2021,
        "https://www.lseg.com/content/dam/lseg/en_us/documents/investor-relations/annual-reports/lseg-annual-report-2021.pdf",
    ),
    (
        "London Stock Exchange Group",
        2022,
        "https://www.lseg.com/content/dam/lseg/en_us/documents/investor-relations/annual-reports/lseg-annual-report-2022.pdf",
    ),
    (
        "London Stock Exchange Group",
        2023,
        "https://www.lseg.com/content/dam/lseg/en_us/documents/investor-relations/annual-reports/lseg-annual-report-2023.pdf",
    ),
    (
        "London Stock Exchange Group",
        2024,
        "https://www.lseg.com/content/dam/lseg/en_us/documents/investor-relations/annual-reports/lseg-annual-report-2024.pdf",
    ),
    (
        "London Stock Exchange Group",
        2025,
        "https://www.lseg.com/content/dam/lseg/en_us/documents/investor-relations/annual-reports/lseg-annual-report-2025.pdf",
    ),
    (
        "Marks and Spencer Group",
        2023,
        "https://corporate.marksandspencer.com/sites/marksandspencer/files/2023-06/M%26S_2023_Annual_Report.pdf",
    ),
    (
        "Merck & Co",
        2025,
        "https://www.merck.com/wp-content/uploads/sites/124/2026/02/MRK-12.31.2025-10K-FINAL.pdf",
    ),
    (
        "NN Group",
        2021,
        "https://www.nn-group.com/site/binaries/content/assets/nn-group/annual-reports/2021/nn-group-annual-report-2021.pdf",
    ),
    (
        "NN Group",
        2022,
        "https://www.nn-group.com/site/binaries/content/assets/nn-group/annual-reports/2022/nn-group-annual-report-2022.pdf",
    ),
    (
        "NN Group",
        2023,
        "https://www.nn-group.com/site/binaries/content/assets/nn-group/annual-reports/2023/nn-group-annual-report-2023.pdf",
    ),
    (
        "NN Group",
        2024,
        "https://www.nn-group.com/site/binaries/content/assets/nn-group/annual-reports/2024/nn-group-annual-report-2024.pdf",
    ),
    (
        "NN Group",
        2025,
        "https://www.nn-group.com/site/binaries/content/assets/nn-group/annual-reports/2025/nn-group-annual-report-2025.pdf",
    ),
    (
        "Nokia",
        2019,
        "https://www.nokia.com/system/files/2020-03/Nokia%20in%202019%20annual%20report_1.pdf",
    ),
    (
        "Nokia",
        2020,
        "https://www.nokia.com/system/files/2021-03/Nokia_Annual_Report_2020_English.pdf",
    ),
    ("Nokia", 2022, "https://www.nokia.com/system/files/2022-03/nokia-ar21-en.pdf"),
    ("Nokia", 2022, "https://www.nokia.com/system/files/2023-03/nokia-annual-report-2022.pdf"),
    ("Nokia", 2023, "https://www.nokia.com/system/files/2024-03/nokia-annual-report-2023.pdf"),
    ("Nokia", 2024, "https://www.nokia.com/system/files/2025-04/nokia-annual-report-2024_1.pdf"),
    ("Nokia", 2025, "https://www.nokia.com/system/files/2026-03/nokia-annual-report-2025.pdf"),
    (
        "Northern Trust",
        2025,
        "https://www.northerntrust.com/content/dam/northerntrust/corporate/global/en/documents/about-us/investor-relations/2025/northern-trust-2025-10-k.pdf",
    ),
    (
        "Novo Nordisk",
        2021,
        "https://www.novonordisk.com/content/dam/nncorp/global/en/investors/irmaterial/annual_report/2022/novo-nordisk-annual-report-2021.pdf",
    ),
    (
        "Novo Nordisk",
        2022,
        "https://www.novonordisk.com/content/dam/nncorp/global/en/investors/irmaterial/annual_report/2023/novo-nordisk-annual-report-2022.pdf",
    ),
    (
        "Novo Nordisk",
        2023,
        "https://www.novonordisk.com/content/dam/nncorp/global/en/investors/irmaterial/annual_report/2024/novo-nordisk-annual-report-2023.pdf",
    ),
    (
        "Novo Nordisk",
        2024,
        "https://www.novonordisk.com/content/dam/nncorp/global/en/investors/irmaterial/annual_report/2025/novo-nordisk-annual-report-2024.pdf",
    ),
    (
        "Novo Nordisk",
        2025,
        "https://www.novonordisk.com/content/dam/nncorp/global/en/investors/irmaterial/annual_report/2026/novo-nordisk-annual-report-2025.pdf",
    ),
    ("Phoenix Group", 2019, "https://library.standardlife.co.uk/annual-report-2019.pdf"),
    (
        "Phoenix Group",
        2022,
        "https://library.standardlife.co.uk/phoenix-group-annual-report-and-accounts-2022.pdf",
    ),
    (
        "Phoenix Group",
        2023,
        "https://library.standardlife.co.uk/phoenix-group-annual-report-and-accounts-2023.pdf",
    ),
    (
        "Phoenix Group",
        2024,
        "https://library.standardlifeplc.com/phoenix_group_annual_report_and_accounts_2024.pdf",
    ),
    (
        "Phoenix Group",
        2025,
        "https://library.standardlife.co.uk/annual-report-and-accounts-2025-spread.pdf",
    ),
    (
        "Plus500",
        2024,
        "https://cdn.plus500.com/Media/Investors/Reports/Plus500_FY_Annual_Report_2024.pdf",
    ),
    (
        "Plus500",
        2025,
        "https://cdn.plus500.com/Media/Investors/Reports/Plus500_FY_Annual_Report_2025.pdf",
    ),
    (
        "Procter & Gamble",
        2023,
        "https://s204.q4cdn.com/332108499/files/doc_financials/2023/ar/f3672cf9-5f0c-4352-aa0f-044f80466073.pdf",
    ),
    (
        "RELX",
        2021,
        "https://www.relx.com/~/media/Files/R/RELX-Group/documents/reports/annual-reports/relx-2021-annual-report.pdf",
    ),
    (
        "RELX",
        2022,
        "https://www.relx.com/~/media/Files/R/RELX-Group/documents/reports/annual-reports/relx-2022-annual-report.pdf",
    ),
    (
        "RELX",
        2023,
        "https://www.relx.com/~/media/Files/R/RELX-Group/documents/reports/annual-reports/relx-2023-annual-report.pdf",
    ),
    (
        "RELX",
        2024,
        "https://www.relx.com/~/media/Files/R/RELX-Group/documents/reports/annual-reports/relx-2024-annual-report.pdf",
    ),
    (
        "RELX",
        2025,
        "https://www.relx.com/~/media/Files/R/RELX-Group/documents/reports/annual-reports/relx-2025-annual-report.pdf",
    ),
    (
        "Royal Bank of Canada",
        2019,
        "https://www.rbc.com/investor-relations/_assets-custom/pdf/ar_2019_e.pdf",
    ),
    (
        "Royal Bank of Canada",
        2020,
        "https://www.rbc.com/investor-relations/_assets-custom/pdf/ar_2020_e.pdf",
    ),
    (
        "Royal Bank of Canada",
        2021,
        "https://www.rbc.com/investor-relations/_assets-custom/pdf/ar_2021_e.pdf",
    ),
    (
        "Royal Bank of Canada",
        2022,
        "https://www.rbc.com/investor-relations/_assets-custom/pdf/ar_2022_e.pdf",
    ),
    (
        "Royal Bank of Canada",
        2023,
        "https://www.rbc.com/investor-relations/_assets-custom/pdf/ar_2023_e.pdf",
    ),
    (
        "Royal Bank of Canada",
        2024,
        "https://www.rbc.com/investor-relations/_assets-custom/pdf/ar_2024_e.pdf",
    ),
    (
        "Royal Bank of Canada",
        2025,
        "https://www.rbc.com/investor-relations/_assets-custom/pdf/ar_2025_e.pdf",
    ),
    (
        "Sage Group",
        2020,
        "https://www.sage.com/investors/-/media/files/investors/documents/pdf/annual%20report/annual%20report%202020.pdf",
    ),
    (
        "Sage Group",
        2021,
        "https://www.sage.com/investors/-/media/files/investors/documents/pdf/annual%20report/annual-report-2021-2.pdf",
    ),
    (
        "Sage Group",
        2022,
        "https://www.sage.com/investors/-/media/files/investors/documents/pdf/annual%20report/sage-annual-report-2022.pdf",
    ),
    (
        "Sage Group",
        2023,
        "https://www.sage.com/investors/-/media/files/investors/documents/pdf/annual%20report/sage-annual-report-2023.pdf",
    ),
    (
        "Sage Group",
        2024,
        "https://www.sage.com/investors/-/media/files/investors/documents/pdf/annual%20report/sage-annual-report-2024.pdf",
    ),
    (
        "Sage Group",
        2025,
        "https://www.sage.com/investors/-/media/files/investors/documents/pdf/annual%20report/sage-annual-report-2025.pdf",
    ),
    (
        "Sanofi",
        2019,
        "https://www.sanofi.com/assets/dotcom/content-app/publications/annual-report-on-form-20-f/2019-01-01-form-20-f-2019-en.pdf",
    ),
    (
        "Sanofi",
        2019,
        "https://www.sanofi.com/assets/dotcom/content-app/publications/integrated-report/2019-01-01-sanofi-integrated-report-en.pdf",
    ),
    (
        "Sanofi",
        2020,
        "https://www.sanofi.com/assets/dotcom/content-app/publications/annual-report-on-form-20-f/2020-01-01-form-20-f-2020-en.pdf",
    ),
    (
        "Sanofi",
        2020,
        "https://www.sanofi.com/assets/dotcom/content-app/publications/integrated-report/2020-01-01-sanofi-Integrated-report-en.pdf",
    ),
    (
        "Sanofi",
        2021,
        "https://www.sanofi.com/assets/dotcom/content-app/publications/annual-report-on-form-20-f/2021-01-01-form-20-f-2021-en.pdf",
    ),
    (
        "Sanofi",
        2022,
        "https://www.sanofi.com/assets/dotcom/content-app/publications/annual-report-on-form-20-f/2022-01-01-form-20-f-2022-en.pdf",
    ),
    (
        "Sanofi",
        2022,
        "https://www.sanofi.com/assets/dotcom/content-app/publications/integrated-report/SANOFI-Integrated-Annual-Report-2022-EN.pdf",
    ),
    (
        "Sanofi",
        2023,
        "https://www.sanofi.com/assets/dotcom/content-app/publications/annual-report-on-form-20-f/2023-01-01-form-20-f-2023-en.pdf",
    ),
    (
        "Sanofi",
        2024,
        "https://www.sanofi.com/assets/dotcom/content-app/publications/annual-report-on-form-20-f/2024-01-01-form-20-f-2024-en.pdf",
    ),
    (
        "Sanofi",
        2025,
        "https://www.sanofi.com/assets/dotcom/content-app/publications/annual-report-on-form-20-f/2025-01-01-form-20-f-2025-en.pdf",
    ),
    (
        "Scotiabank",
        2021,
        "https://www.scotiabank.com/content/dam/scotiabank/corporate/quarterly-reports/2021/q4/Annual_Report_2021_EN.pdf",
    ),
    (
        "Scotiabank",
        2022,
        "https://www.scotiabank.com/content/dam/scotiabank/corporate/quarterly-reports/2022/q4/Annual_Report_2022_EN.pdf",
    ),
    (
        "Scotiabank",
        2023,
        "https://www.scotiabank.com/content/dam/scotiabank/corporate/quarterly-reports/2023/q4/Annual_Report_2023_EN.pdf",
    ),
    (
        "Scotiabank",
        2024,
        "https://www.scotiabank.com/content/dam/scotiabank/corporate/quarterly-reports/2024/q4/Annual_Report_2024_EN.pdf",
    ),
    (
        "Scotiabank",
        2025,
        "https://www.scotiabank.com/content/dam/scotiabank/corporate/quarterly-reports/2025/q4/Annual_Report_2025_EN.pdf",
    ),
    (
        "Siemens",
        2019,
        "https://assets.ctfassets.net/17si5cpawjzf/1GN3aRRCXGJ89xipNg1B9U/38f93cae22abff2ade0a47cae3fddad5/siemens-ar2019.pdf",
    ),
    (
        "Siemens",
        2020,
        "https://assets.ctfassets.net/17si5cpawjzf/5Z7OMhTPgxp2x7lBtLRt3S/fb2c2f1d07bc250744d9358f6c340c9a/siemens-ar2020.pdf",
    ),
    (
        "Siemens",
        2021,
        "https://assets.ctfassets.net/17si5cpawjzf/6XxK1XVHXisMaxRS2HcmGZ/2492f75bb45a6b7e085661643277d41b/siemens-ar-2021.pdf",
    ),
    (
        "Siemens",
        2022,
        "https://assets.ctfassets.net/17si5cpawjzf/2OlK51tFTifwxVlYAh3Do2/41152408ccdf2c3efeca49f5170b31d7/Siemens-Annual-Report-2022.pdf",
    ),
    (
        "Siemens",
        2023,
        "https://assets.ctfassets.net/17si5cpawjzf/75NjPd0MkpOq6dkbksOUYX/2d0b2a1263c51e853fcdfc91025dda82/Siemens-Annual-Report-2023.pdf",
    ),
    (
        "Siemens",
        2024,
        "https://assets.new.siemens.com/siemens/assets/api/uuid:344347ec-a1bd-44cb-aaaa-711d1b3ec1b8/Siemens-Annual-Report-2024.pdf",
    ),
    (
        "Siemens",
        2025,
        "https://assets.ctfassets.net/17si5cpawjzf/3XulIKlbpUZ6RplTRZbdKQ/2228242f3c9337073601ddd9e860d36a/Siemens_Report_FY2025__1_.pdf",
    ),
    (
        "Standard Chartered",
        2019,
        "https://www.sc.com/en/uploads/sites/66/content/docs/standard-chartered-plc-full-year-2019-report.pdf",
    ),
    (
        "Standard Chartered",
        2020,
        "https://www.sc.com/en/uploads/sites/66/content/docs/standard-chartered-plc-full-year-2020-report.pdf",
    ),
    (
        "Standard Chartered",
        2021,
        "https://www.sc.com/en/uploads/sites/66/content/docs/standard-chartered-plc-full-year-2021-report.pdf",
    ),
    (
        "Standard Chartered",
        2022,
        "https://www.sc.com/en/uploads/sites/66/content/docs/sc-bank-2022-annual-report.pdf",
    ),
    (
        "Standard Chartered",
        2022,
        "https://www.sc.com/en/uploads/sites/66/content/docs/standard-chartered-plc-full-year-2022-report.pdf",
    ),
    (
        "Standard Chartered",
        2023,
        "https://www.sc.com/en/uploads/sites/66/content/docs/sc-bank-2023-annual-report.pdf",
    ),
    (
        "Standard Chartered",
        2023,
        "https://www.sc.com/en/uploads/sites/66/content/docs/standard-chartered-plc-full-year-2023-report.pdf",
    ),
    (
        "Standard Chartered",
        2024,
        "https://www.sc.com/en/uploads/sites/66/content/docs/standard-chartered-plc-full-year-2024-report.pdf",
    ),
    (
        "Standard Chartered",
        2025,
        "https://www.sc.com/en/uploads/sites/66/content/docs/standard-chartered-plc-2025-annual-report.pdf",
    ),
    (
        "Swiss Re",
        2019,
        "https://www.swissre.com/dam/jcr:d8b8d110-0394-4f0e-8f12-56ce87517f24/2019-annual-srcml-report-doc.2020-09-28-04-18-52.pdf",
    ),
    (
        "Swiss Re",
        2019,
        "https://www.swissre.com/dam/jcr:f5673e2b-a4fd-498c-b2a7-ea2d153715f9/2019-annual-srz-report-doc.2020-09-28-04-18-52.pdf",
    ),
    (
        "Swiss Re",
        2021,
        "https://www.swissre.com/dam/jcr:00f2f0c9-4884-4fcd-be89-5241a51ce6c7/2021-annual-srz-report-doc.2022-03-28-15-54-57.pdf",
    ),
    (
        "Swiss Re",
        2021,
        "https://www.swissre.com/dam/jcr:638184a5-2a5d-41ff-8f25-612a66d4a25b/swiss-re-capital-markets-limited-annual-report-2021.2022-05-09-08-30-27.pdf",
    ),
    (
        "Swiss Re",
        2022,
        "https://www.swissre.com/dam/jcr:59f6bacb-092f-4cb4-aa22-043b2c1a0b26/2022-annual-srz-report-doc.2023-03-15-17-20-46.pdf",
    ),
    (
        "Swiss Re",
        2023,
        "https://www.swissre.com/dam/jcr:d519443a-30e4-4966-8913-88eee6eaebb7/2023-annual-srz-report.2024-04-22-08-32-55.pdf",
    ),
    (
        "Swiss Re",
        2024,
        "https://www.swissre.com/dam/jcr:6a616197-c7be-4921-ad69-3618e637aec8/2024-annual-srz-report.2025-04-29-15-33-23.pdf",
    ),
    (
        "Swiss Re",
        2024,
        "https://www.swissre.com/dam/jcr:6d5cbb29-38a1-4b7f-aac2-f2b45eabe1b6/2024-annual-report.2026-02-19-16-07-13.pdf",
    ),
    (
        "Swiss Re",
        2025,
        "https://www.swissre.com/dam/jcr:a7e9dca5-1911-404d-bc9d-a8875ca7cab9/2025-annual-report.2026-03-12-05-19-06.pdf",
    ),
    (
        "Swiss Re",
        2025,
        "https://www.swissre.com/dam/jcr:ac057a57-915f-4769-91df-c32ff3e141e9/2025-annual-report-financial-statements.2026-03-12-05-19-07.pdf",
    ),
    (
        "TD Bank Group",
        2025,
        "https://www.td.com/content/dam/tdcom/canada/about-td/pdf/investor/investor-relations/financial-information/financial-reports/annual-reports/annual-report-2025/ar2025-consolidated-financial-statements-en.pdf",
    ),
    (
        "TD Bank Group",
        2025,
        "https://www.td.com/content/dam/tdcom/canada/about-td/pdf/investor/investor-relations/financial-information/financial-reports/annual-reports/annual-report-2025/ar2025-managements-discussion-and-analysis-en.pdf",
    ),
    (
        "Telstra",
        2019,
        "https://www.telstra.com.au/content/dam/tcom/about-us/investors/pdf%20F/2019-Annual-Report.PDF",
    ),
    (
        "Telstra",
        2020,
        "https://www.telstra.com.au/content/dam/tcom/about-us/investors/pdf%20F/2020-Telstra-Annual-Report.pdf",
    ),
    (
        "Telstra",
        2022,
        "https://www.telstra.com.au/content/dam/tcom/about-us/investors/pdf-g/TEL-AR-2022-Pages-FINAL.pdf",
    ),
    (
        "Telstra",
        2023,
        "https://www.telstra.com.au/content/dam/tcom/about-us/investors/pdf-g/telstra-annual-report-2023.pdf",
    ),
    (
        "Telstra",
        2024,
        "https://www.telstra.com.au/content/dam/tcom/about-us/investors/pdf-g/telstra-annual-report-2024.pdf",
    ),
    (
        "Telstra",
        2025,
        "https://www.telstra.com.au/content/dam/tcom/about-us/investors/pdf-g/telstra-annual-report-2025.pdf",
    ),
    (
        "UnitedHealth Group",
        2020,
        "https://www.unitedhealthgroup.com/content/dam/UHG/PDF/investors/2020/UNH-Form-10-K.pdf",
    ),
    (
        "Verizon",
        2019,
        "https://www.verizon.com/about/sites/default/files/2019-Verizon-Annual-Report.pdf",
    ),
    (
        "Verizon",
        2020,
        "https://www.verizon.com/about/sites/default/files/2020-Annual-Report-on-Form-10-K.PDF",
    ),
    (
        "Verizon",
        2021,
        "https://www.verizon.com/about/sites/default/files/2021-Annual-Report-on-Form-10-K.pdf",
    ),
    (
        "Verizon",
        2022,
        "https://www.verizon.com/about/sites/default/files/2022-Annual-Report-on-Form-10K.pdf",
    ),
    (
        "Verizon",
        2023,
        "https://www.verizon.com/about/sites/default/files/2023-Annual-Report-on-Form-10k.pdf",
    ),
    (
        "Verizon",
        2024,
        "https://www.verizon.com/about/sites/default/files/2024-Annual-Report-on-Form-10k.pdf",
    ),
    (
        "Verizon",
        2025,
        "https://www.verizon.com/about/sites/default/files/2025-Annual-Report-on-Form-10k.pdf",
    ),
    (
        "WPP",
        2025,
        "https://cdn.wpp.com/raw/upload/v1778673483/wpp/project/wpp/files/investors/2026/annual-report/wpp_annual_report_2025.pdf",
    ),
    (
        "Wells Fargo",
        2020,
        "https://www.wellsfargo.com/assets/pdf/about/investor-relations/annual-reports/2020-annual-report.pdf",
    ),
    (
        "Wells Fargo",
        2021,
        "https://www.wellsfargo.com/assets/pdf/about/investor-relations/annual-reports/2021-annual-report.pdf",
    ),
    (
        "Wells Fargo",
        2021,
        "https://www.wellsfargo.com/assets/pdf/about/investor-relations/sec-filings/2021/10k.pdf",
    ),
    (
        "Wells Fargo",
        2022,
        "https://www.wellsfargo.com/assets/pdf/about/investor-relations/annual-reports/2022-annual-report.pdf",
    ),
    (
        "Wells Fargo",
        2022,
        "https://www.wellsfargo.com/assets/pdf/about/investor-relations/sec-filings/2022/10k.pdf",
    ),
    (
        "Wells Fargo",
        2023,
        "https://www.wellsfargo.com/assets/pdf/about/investor-relations/annual-reports/2023-annual-report.pdf",
    ),
    (
        "Wells Fargo",
        2024,
        "https://www.wellsfargo.com/assets/pdf/about/investor-relations/annual-reports/2024-annual-report.pdf",
    ),
    (
        "Wells Fargo",
        2025,
        "https://www.wellsfargo.com/assets/pdf/about/investor-relations/annual-reports/2025-annual-report.pdf",
    ),
    (
        "Westpac",
        2024,
        "https://www.westpac.com.au/content/dam/public/wbc/documents/pdf/aw/ic/wbc-annual-report-2024.pdf",
    ),
    (
        "Westpac",
        2025,
        "https://www.westpac.com.au/content/dam/public/wbc/documents/pdf/aw/ic/wbc-annual-report-2025.pdf",
    ),
    (
        "Woolworths Group",
        2020,
        "https://www.woolworthsgroup.com.au/content/dam/wwg/investors/reports/2020/195794_annual-report-2020.pdf",
    ),
    (
        "Woolworths Group",
        2021,
        "https://www.woolworthsgroup.com.au/content/dam/wwg/investors/reports/2021/195984_annual-report-2021.pdf",
    ),
)


def _candidate(publisher: str, year: int, url: str) -> Candidate:
    tag = hashlib.sha1(url.encode()).hexdigest()[:6]
    return Candidate(
        id=safe_id("ir", f"{publisher}-{year}-{tag}"),
        url=url,
        source=NAME,
        licence=LICENCE,
        doc_type_hint="financial.annual_report",
        title=f"{publisher} {year} annual report",
        publisher=publisher,
        family_hint="annual_report",
        extra={"year": year},
    )


def candidates(net: Net) -> Iterator[Candidate]:
    """No listing request: the URLs above are the list, interleaved by company, newest first."""
    by_publisher: dict[str, list[Candidate]] = defaultdict(list)
    for publisher, year, url in sorted(REPORTS, key=lambda r: (r[0], -r[1], r[2])):
        by_publisher[publisher].append(_candidate(publisher, year, url))
    return round_robin(*(by_publisher[p] for p in sorted(by_publisher)))


SOURCE = Source(
    name=NAME,
    target=250,
    cap=len(REPORTS),
    probe_url=None,  # many hosts: each is checked against its robots.txt per download
    candidates=candidates,
)
