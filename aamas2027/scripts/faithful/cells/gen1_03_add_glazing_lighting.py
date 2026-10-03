epmodel.add_glazing_system(gs_sr2_ec01)
epmodel.add_glazing_system(gs_sr2_ec02)
epmodel.add_glazing_system(gs_sr2_ec03)
epmodel.add_glazing_system(gs_sr2_ec04)
epmodel.add_lighting(
    zone="Perimeter_mid_ZN_1",
    lighting_level=2231, # 
    replace=True
)
epmodel.add_lighting(
    zone="Perimeter_mid_ZN_2",
    lighting_level=2231, # 
    replace=True
)
epmodel.add_lighting(
    zone="Perimeter_mid_ZN_3",
    lighting_level=1412, # 
    replace=True
)
epmodel.add_lighting(
    zone="Perimeter_mid_ZN_4",
    lighting_level=1412, # 
    replace=True
)
epmodel.add_lighting(
    zone="Core_mid",
    lighting_level=10586, # 
    replace=True
)
