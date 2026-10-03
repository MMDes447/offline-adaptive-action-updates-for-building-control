gs_sr2_ec01 = fr.create_glazing_system(
    name="sr2_ec01",
    layer_inputs=[
        fr.LayerInput("glaizng_2/glaizng_2___/sageglass-cooltonetm-sr2-7mm-lami-clear-tint.json"),
        fr.LayerInput("glaizngs/igsdb_product_14028.json"),
    ],
    gaps=[
        fr.Gap(
            gas=[fr.Gas("air", 0.1), fr.Gas("argon", 0.9)],
            thickness_m=0.0127,
        )
    ],
)

gs_sr2_ec02 = fr.create_glazing_system(
    name="sr2_ec02",
    layer_inputs=[
        fr.LayerInput("glaizng_2/glaizng_2___/sageglass-cooltonetm-sr2-7mm-lami-light-tint.json"),
        fr.LayerInput("glaizngs/igsdb_product_14028.json"),
    ],
    gaps=[
        fr.Gap(
            gas=[fr.Gas("air", 0.1), fr.Gas("argon", 0.9)],
            thickness_m=0.0127,
        )
    ],
)

gs_sr2_ec03 = fr.create_glazing_system(
    name="sr2_ec03",
    layer_inputs=[
        fr.LayerInput("glaizng_2/glaizng_2___/sageglass-cooltonetm-sr2-7mm-lami-medium-tint.json"),
        fr.LayerInput("glaizngs/igsdb_product_14028.json"),
    ],
    gaps=[
        fr.Gap(
            gas=[fr.Gas("air", 0.1), fr.Gas("argon", 0.9)],
            thickness_m=0.0127,
        )
    ],
)

gs_sr2_ec04 = fr.create_glazing_system(
    name="sr2_ec04",
    layer_inputs=[
        fr.LayerInput("glaizng_2/glaizng_2___/sageglass-cooltonetm-sr2-7mm-lami-full-tint.json"),
        fr.LayerInput("glaizngs/igsdb_product_14028.json"),
    ],
    gaps=[
        fr.Gap(
            gas=[fr.Gas("air", 0.1), fr.Gas("argon", 0.9)],
            thickness_m=0.0127,
        )
    ],
)

