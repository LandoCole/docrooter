"""Fund documents from public fund-sponsor URLs, listed explicitly (spec §5, controller ruling).

EDGAR has fund documents only as HTML (research brief §3.1), so PDFs come from the sponsors' own
public literature URLs, listed below in full; nothing is discovered by crawling. Sponsors were
chosen because their robots.txt allows these paths to all agents (checked 2026-10-19):

- State Street (SSGA) SPDR ETF fact sheets, one per ticker in SSGA's public "SPDR ETF listing"
  (target-maturity muni series left out);
- iShares (BlackRock) summary prospectuses, one per fund, the link read from the fund's product
  page (whose URL comes from iShares' public product-screener listing);
- iShares annual and semi-annual tailored shareholder reports, per ticker.

Each URL answered ``206 application/pdf`` to a one-kilobyte range request on 2026-10-19
(SSGA's QNDX fact sheet, a 404, was removed). The documents belong to their sponsors: they are
kept locally for research and **never redistributed**; the manifest holds only URLs and hashes.
The four lists are interleaved so any quota draws a mix of sponsors and document kinds.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from benchmarks.training_corpus._net import Net
from benchmarks.training_corpus.sources.base import Candidate, Source, round_robin

NAME = "fund_docs"
LICENCE = "fund-sponsor public document; local research use only, never redistributed"

SSGA_FACT_SHEETS: tuple[str, ...] = (
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-allw.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-bil.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-bils.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-bwx.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-bwz.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-cery.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-cnrg.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-cwb.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-cwi.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-deco.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-dgt.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-dia.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-dwx.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-ebnd.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-ediv.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-eemx.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-efax.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-efiv.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-emhc.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-ewx.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-fez.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-fisr.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-flrn.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-gal.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-gii.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-gmf.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-gnr.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-gwx.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-gxc.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-hail.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-heco.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-hybl.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-hymb.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-ibnd.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-inkm.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-jnk.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-kbe.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-kce.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-kie.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-komp.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-kre.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-lglv.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-lvln.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-mbnd.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-mdy.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-mdyg.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-mdyv.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-mmk.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-mmtm.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-nanr.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-nzac.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-obnd.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-oneo.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-onev.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-oney.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-prab.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-priv.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-prsd.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-psk.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-qefa.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-qemm.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-qus.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-qwld.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-rly.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-rokt.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-rwo.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-rwr.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-rwx.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-sdy.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-she.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-shm.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-sims.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-sjnk.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-slyg.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-slyv.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-smlv.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spab.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spbo.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spdg.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spdw.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spem.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-speu.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spgm.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-sphy.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spib.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spin.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spip.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-splb.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spmb.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spmd.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spsb.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spsm.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-sptb.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spti.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-sptl.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-sptm.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spts.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-sptu.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spy.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spyd.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spyg.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spym.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spyv.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spyx.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-srln.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-stot.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-tekx.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-tfi.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-tipx.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-totl.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-ulst.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-vlu.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-wdiv.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-wip.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xar.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xbi.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xcny.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xes.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xhb.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xhe.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xhs.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xitk.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlb.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlbi.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlc.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlci.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xle.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlei.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlf.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlfi.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xli.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlii.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlk.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlki.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlp.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlre.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlri.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlsi.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlsr.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlu.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlui.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlv.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlvi.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xly.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xlyi.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xme.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xntk.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xop.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xph.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xrt.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xsd.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xsw.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xtl.pdf",
    "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-xtn.pdf",
)

ISHARES_ANNUAL_REPORTS: tuple[str, ...] = (
    "https://www.ishares.com/us/literature/annual-report/ar-ivv-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ijh-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ijr-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-itot-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iwm-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iwb-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iwf-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iwd-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iwo-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iwn-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iwr-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iwv-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-efa-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-eem-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iefa-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iemg-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-agg-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iusb-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-tlt-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ief-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-shy-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iei-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-tip-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-lqd-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-hyg-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-mbb-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-mub-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-emb-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-igsb-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-igib-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ushy-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-shv-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-govt-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-acwi-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-acwx-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ewj-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ewz-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ewg-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ewu-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ewc-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ewa-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ewy-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ewt-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-fxi-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-mchi-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-inda-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ibb-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iyr-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iyw-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-iyf-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ita-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-itb-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-soxx-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-igv-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-usmv-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-efav-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-mtum-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-qual-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-dvy-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-hdv-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-idv-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-scz-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-ixus-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-flot-en.pdf",
    "https://www.ishares.com/us/literature/annual-report/ar-stip-en.pdf",
)

ISHARES_SEMIANNUAL_REPORTS: tuple[str, ...] = (
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ivv-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ijh-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ijr-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-itot-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iwm-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iwb-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iwf-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iwd-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iwo-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iwn-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iwr-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iwv-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-efa-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-eem-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iefa-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iemg-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-agg-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iusb-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-tlt-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ief-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-shy-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iei-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-tip-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-lqd-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-hyg-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-mbb-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-mub-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-emb-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-igsb-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-igib-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ushy-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-shv-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-govt-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-acwi-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-acwx-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ewj-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ewz-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ewg-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ewu-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ewc-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ewa-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ewy-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ewt-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-fxi-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-mchi-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-inda-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ibb-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iyr-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iyw-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-iyf-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ita-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-itb-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-soxx-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-igv-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-usmv-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-efav-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-mtum-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-qual-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-dvy-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-hdv-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-idv-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-scz-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-ixus-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-flot-en.pdf",
    "https://www.ishares.com/us/literature/semi-annual-report/sar-stip-en.pdf",
)


# (ticker, URL): each fund's summary prospectus link, read from its iShares product page.
ISHARES_SUMMARY_PROSPECTUSES: tuple[tuple[str, str], ...] = (
    (
        "ivv",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-core-s-and-p-500-etf-3-31.pdf",
    ),
    (
        "ijh",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-core-s-and-p-mid-cap-etf-3-31.pdf",
    ),
    (
        "ijr",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-core-s-and-p-small-cap-etf-3-31.pdf",
    ),
    (
        "itot",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-core-s-and-p-total-us-stock-market-etf-3-31.pdf",
    ),
    (
        "iwm",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-russell-2000-etf-3-31.pdf",
    ),
    (
        "iwb",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-russell-1000-etf-3-31.pdf",
    ),
    (
        "iwf",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-russell-1000-growth-etf-3-31.pdf",
    ),
    (
        "iwd",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-russell-1000-value-etf-3-31.pdf",
    ),
    (
        "iwo",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-russell-2000-growth-etf-3-31.pdf",
    ),
    (
        "iwn",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-russell-2000-value-etf-3-31.pdf",
    ),
    (
        "iwr",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-russell-mid-cap-etf-3-31.pdf",
    ),
    (
        "iwv",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-russell-3000-etf-3-31.pdf",
    ),
    (
        "efa",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-eafe-etf-7-31.pdf",
    ),
    (
        "eem",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-emerging-markets-etf-8-31.pdf",
    ),
    (
        "iefa",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-core-msci-eafe-etf-7-31.pdf",
    ),
    (
        "iemg",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-core-msci-emerging-markets-etf-8-31.pdf",
    ),
    (
        "agg",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-core-us-aggregate-bond-etf-2-28.pdf",
    ),
    (
        "iusb",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-core-total-usd-bond-market-etf-10-31.pdf",
    ),
    (
        "tlt",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-20-plus-year-treasury-bond-etf-2-28.pdf",
    ),
    (
        "ief",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-7-10-year-treasury-bond-etf-2-28.pdf",
    ),
    (
        "shy",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-1-3-year-treasury-bond-etf-2-28.pdf",
    ),
    (
        "iei",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-3-7-year-treasury-bond-etf-2-28.pdf",
    ),
    (
        "tip",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-tips-bond-etf-10-31.pdf",
    ),
    (
        "lqd",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-iboxx-investment-grade-corporate-bond-etf-2-28.pdf",
    ),
    (
        "hyg",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-iboxx-high-yield-corporate-bond-etf-2-28.pdf",
    ),
    ("mbb", "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-mbs-etf-2-28.pdf"),
    (
        "mub",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-national-muni-bond-etf-2-28.pdf",
    ),
    (
        "emb",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-jp-morgan-usd-emerging-markets-bond-etf-10-31.pdf",
    ),
    (
        "igsb",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-1-3-year-credit-bond-etf-2-28.pdf",
    ),
    (
        "igib",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-intermediate-credit-bond-etf-2-28.pdf",
    ),
    (
        "ushy",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-broad-usd-high-yield-corporate-bond-etf-10-31.pdf",
    ),
    (
        "shv",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-short-treasury-bond-etf-2-28.pdf",
    ),
    (
        "govt",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-core-us-treasury-bond-etf-10-31.pdf",
    ),
    (
        "acwi",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-acwi-etf-7-31.pdf",
    ),
    (
        "acwx",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-acwi-ex-us-etf-7-31.pdf",
    ),
    (
        "ewj",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-japan-etf-8-31.pdf",
    ),
    (
        "ewz",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-brazil-capped-etf-8-31.pdf",
    ),
    (
        "ewg",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-germany-etf-8-31.pdf",
    ),
    (
        "ewu",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-united-kingdom-etf-8-31.pdf",
    ),
    (
        "ewc",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-canada-etf-8-31.pdf",
    ),
    (
        "ewa",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-australia-etf-8-31.pdf",
    ),
    (
        "ewy",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-south-korea-capped-etf-8-31.pdf",
    ),
    (
        "ewt",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-taiwan-capped-etf-8-31.pdf",
    ),
    (
        "fxi",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-china-large-cap-etf-7-31.pdf",
    ),
    (
        "mchi",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-china-etf-8-31.pdf",
    ),
    (
        "inda",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-india-etf-8-31.pdf",
    ),
    (
        "ibb",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-nasdaq-biotechnology-etf-3-31.pdf",
    ),
    (
        "iyr",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-us-real-estate-etf-3-31.pdf",
    ),
    (
        "iyw",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-us-technology-etf-4-30.pdf",
    ),
    (
        "iyf",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-us-financials-etf-4-30.pdf",
    ),
    (
        "ita",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-us-aerospace-and-defense-etf-3-31.pdf",
    ),
    (
        "itb",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-us-home-construction-etf-3-31.pdf",
    ),
    (
        "soxx",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-phlx-semiconductor-etf-3-31.pdf",
    ),
    (
        "igv",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-north-american-tech-software-etf-3-31.pdf",
    ),
    (
        "usmv",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-edge-msci-min-vol-usa-etf-7-31.pdf",
    ),
    (
        "efav",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-edge-msci-min-vol-eafe-etf-7-31.pdf",
    ),
    (
        "mtum",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-edge-msci-usa-momentum-factor-etf-7-31.pdf",
    ),
    (
        "qual",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-edge-msci-usa-quality-factor-etf-7-31.pdf",
    ),
    (
        "dvy",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-select-dividend-etf-4-30.pdf",
    ),
    (
        "hdv",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-core-high-dividend-etf-4-30.pdf",
    ),
    (
        "idv",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-international-select-dividend-etf-4-30.pdf",
    ),
    (
        "scz",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-msci-eafe-small-cap-etf-7-31.pdf",
    ),
    (
        "ixus",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-core-msci-total-international-stock-etf-7-31.pdf",
    ),
    (
        "flot",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-floating-rate-bond-etf-10-31.pdf",
    ),
    (
        "stip",
        "https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-0-5-year-tips-bond-etf-10-31.pdf",
    ),
)


def _ticker(url: str, pattern: str) -> str:
    match = re.search(pattern, url)
    if match is None:
        raise ValueError(f"unexpected fund document URL: {url}")
    return match.group(1)


def _ssga(url: str) -> Candidate:
    ticker = _ticker(url, r"factsheet-us-en-([a-z0-9]+)\.pdf$")
    return Candidate(
        id=f"fund-ssga-fs-{ticker}",
        url=url,
        source=NAME,
        licence=LICENCE,
        doc_type_hint="financial.fact_sheet",
        title=f"SPDR {ticker.upper()} fact sheet",
        publisher="State Street Global Advisors",
        family_hint="fund_document",
        extra={"sponsor": "SSGA", "ticker": ticker.upper(), "kind": "fact sheet"},
    )


def _ishares(url: str, kind: str) -> Candidate:
    prefix = "ar" if kind == "annual" else "sar"
    ticker = _ticker(url, rf"/{prefix}-([a-z0-9]+)-en\.pdf$")
    return Candidate(
        id=f"fund-ishares-{prefix}-{ticker}",
        url=url,
        source=NAME,
        licence=LICENCE,
        # A fund's shareholder report: the annual-report type, in the fund-document family.
        doc_type_hint="financial.annual_report",
        title=f"iShares {ticker.upper()} {kind} shareholder report",
        publisher="BlackRock (iShares)",
        family_hint="fund_document",
        extra={"sponsor": "iShares", "ticker": ticker.upper(), "kind": f"{kind} report"},
    )


def _ishares_summary_prospectus(ticker: str, url: str) -> Candidate:
    return Candidate(
        id=f"fund-ishares-sp-{ticker}",
        url=url,
        source=NAME,
        licence=LICENCE,
        doc_type_hint="financial.prospectus",
        title=f"iShares {ticker.upper()} summary prospectus",
        publisher="BlackRock (iShares)",
        family_hint="fund_document",
        extra={"sponsor": "iShares", "ticker": ticker.upper(), "kind": "summary prospectus"},
    )


def candidates(net: Net) -> Iterator[Candidate]:
    """No listing request: the URLs above are the list."""
    return round_robin(
        (_ssga(u) for u in SSGA_FACT_SHEETS),
        (_ishares_summary_prospectus(t, u) for t, u in ISHARES_SUMMARY_PROSPECTUSES),
        (_ishares(u, "annual") for u in ISHARES_ANNUAL_REPORTS),
        (_ishares(u, "semi-annual") for u in ISHARES_SEMIANNUAL_REPORTS),
    )


SOURCE = Source(
    name=NAME,
    target=150,
    cap=len(SSGA_FACT_SHEETS)
    + len(ISHARES_SUMMARY_PROSPECTUSES)
    + len(ISHARES_ANNUAL_REPORTS)
    + len(ISHARES_SEMIANNUAL_REPORTS),
    probe_url="https://www.ssga.com/library-content/products/factsheets/",
    candidates=candidates,
    note="iShares URLs are checked against www.ishares.com robots.txt per download",
)
