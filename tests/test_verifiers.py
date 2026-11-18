"""Unit tests for deterministic engineering code-gate verifiers."""

import unittest

from harness.verifiers import (
    verify_cooling_tower_performance,
    verify_nfpa92_stairwell_pressurization,
    verify_open_channel_trapezoidal_flow,
    verify_weir_discharge_capacity,
    verify_water_hammer_surge,
    verify_infinite_slope_stability,
    verify_asme_bpvc_vessel_thickness,
    verify_heat_exchanger_lmtd,
    verify_open_channel_hydraulic_jump,
    verify_nrcs_tr55_runoff,
    verify_asce7_snow_drift,
    verify_aisc_block_shear,
    verify_nfpa13_fire_sprinkler_demand,
    verify_asce7_nonstructural_component_seismic,
    verify_nec_equipment_grounding_conductor,
    verify_aashto_horizontal_curve,
    verify_meyerhof_bearing_capacity,
    verify_asme_b313_pipe_thickness,
    verify_asce7_seismic_force_distribution,
    verify_aisc_web_local_yielding_and_crippling,
    verify_aisc_unbraced_beam_flexure,
    verify_aashto_stopping_sight_distance,
    verify_nec_grounding_electrode_conductor,
    verify_asce7_story_drift_and_pdelta,
    verify_darcy_weisbach_pipe_flow,
    verify_aci_column_axial_load,
    verify_aisc_beam_column_interaction,
    verify_asce7_roof_snow_load,
    verify_aci_rebar_development_length,
    verify_aisc_column_base_plate,
    verify_hvac_cooling_load_capacity,
    verify_aci_punching_shear,
    verify_stormwater_rational_runoff,
    verify_nec_motor_branch_circuit,
    verify_clay_consolidation_settlement,
    verify_aisc_tension_member,
    verify_nec_conduit_fill,
    verify_aisc_fillet_weld,
    verify_nec_solar_busbar_interconnection,
    verify_asce7_wind_pressure,
    verify_pump_npsh,
    verify_aisc_bolted_connection,
    verify_ashrae_ventilation_rate,
    verify_asce7_seismic_base_shear,
    verify_hydronic_pipe_sizing,
    verify_transformer_fault_current,
    verify_nec_conductor_ampacity,
    verify_manning_pipe_flow,
    verify_aci_beam_shear,
    verify_retaining_wall_stability,
    verify_aci_beam_flexure,
    verify_ashrae_duct_sizing,
    verify_aisc_column_compression,
    check_aabb_clashes,
    rank_project_files,
    verify_aisc_w_beam,
    verify_footing_bearing_and_settlement,
    verify_nec_voltage_drop,
)


class TestEngineeringVerifiers(unittest.TestCase):
    def test_revision_authority_ordering(self) -> None:
        files = [
            "01_Preliminary_Layout.pdf",
            "02_IFC_Structural.pdf",
            "03_Addendum_02_Elec.pdf",
            "04_AHJ_Letter_Approval.pdf",
        ]
        ranked = rank_project_files(files)
        self.assertEqual(ranked[0]["file"], "04_AHJ_Letter_Approval.pdf")
        self.assertEqual(ranked[1]["file"], "03_Addendum_02_Elec.pdf")
        self.assertEqual(ranked[-1]["file"], "01_Preliminary_Layout.pdf")

    def test_aisc_w_beam_pass_and_fail(self) -> None:
        ok = verify_aisc_w_beam(
            span_ft=20.0, w_dead_klf=0.5, w_live_klf=1.0,
            zx_in3=64.0, ix_in4=518.0, d_in=15.9, tw_in=0.295,
        )
        self.assertTrue(ok["pass"])
        bad = verify_aisc_w_beam(
            span_ft=40.0, w_dead_klf=2.0, w_live_klf=3.0,
            zx_in3=30.0, ix_in4=200.0, d_in=12.0, tw_in=0.20,
        )
        self.assertFalse(bad["pass"])

    def test_footing_and_nec(self) -> None:
        geo = verify_footing_bearing_and_settlement(2.5, 8.0, 0.65)
        self.assertTrue(geo["pass"])
        elec = verify_nec_voltage_drop(current_a=40.0, length_ft=150.0, cmil=26240.0, voltage_v=480.0)
        self.assertTrue(elec["pass"])

    def test_aabb_clash_detection(self) -> None:
        boxes = [
            ("duct_1", (0.0, 0.0, 0.0), (10.0, 2.0, 2.0)),
            ("pipe_1", (5.0, 1.0, 1.0), (6.0, 5.0, 3.0)),
            ("conduit_1", (20.0, 20.0, 0.0), (22.0, 22.0, 1.0)),
        ]
        clashes = check_aabb_clashes(boxes)
        self.assertEqual(clashes, [("duct_1", "pipe_1")])


    def test_aisc_column_compression(self) -> None:
        from harness.verifiers import verify_aisc_column_compression

        res_pass = verify_aisc_column_compression(
            p_dead_kips=200.0,
            p_live_kips=400.0,
            kl_x_ft=15.0,
            kl_y_ft=15.0,
            rx_in=6.14,
            ry_in=3.70,
            ag_in2=26.5,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertAlmostEqual(res_pass["kl_r"], (15.0 * 12.0) / 3.70, places=3)

        res_fail = verify_aisc_column_compression(
            p_dead_kips=800.0,
            p_live_kips=1000.0,
            kl_x_ft=40.0,
            kl_y_ft=40.0,
            rx_in=6.14,
            ry_in=3.70,
            ag_in2=26.5,
        )
        self.assertFalse(res_fail["pass"])

        res_edge = verify_aisc_column_compression(
            p_dead_kips=10.0,
            p_live_kips=10.0,
            kl_x_ft=-5.0,
            kl_y_ft=10.0,
            rx_in=1.0,
            ry_in=1.0,
            ag_in2=5.0,
        )
        self.assertFalse(res_edge["pass"])
        self.assertEqual(res_edge["reason"], "non_positive_geometry_or_material")


    def test_ashrae_duct_sizing(self) -> None:
        from harness.verifiers import verify_ashrae_duct_sizing

        res_pass = verify_ashrae_duct_sizing(
            airflow_cfm=1200.0,
            width_in=20.0,
            height_in=12.0,
            max_velocity_fpm=1200.0,
            max_pressure_drop_per_100ft=0.10,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["velocity_fpm"], 1200.0)
        self.assertLess(res_pass["pressure_drop_per_100ft"], 0.10)
        self.assertAlmostEqual(res_pass["aspect_ratio"], 20.0 / 12.0, places=2)

        res_fail_vel = verify_ashrae_duct_sizing(
            airflow_cfm=3000.0,
            width_in=12.0,
            height_in=10.0,
            max_velocity_fpm=1500.0,
            max_pressure_drop_per_100ft=0.10,
        )
        self.assertFalse(res_fail_vel["pass"])
        self.assertGreater(res_fail_vel["velocity_dcr"], 1.0)

        res_fail_ar = verify_ashrae_duct_sizing(
            airflow_cfm=500.0,
            width_in=40.0,
            height_in=6.0,
            max_aspect_ratio=4.0,
        )
        self.assertFalse(res_fail_ar["pass"])
        self.assertGreater(res_fail_ar["aspect_ratio"], 4.0)

        res_nan = verify_ashrae_duct_sizing(
            airflow_cfm=float("nan"),
            width_in=12.0,
            height_in=12.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_ashrae_duct_sizing(
            airflow_cfm=-500.0,
            width_in=12.0,
            height_in=12.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_geometry_or_airflow")


    def test_aci_beam_flexure(self) -> None:
        from harness.verifiers import verify_aci_beam_flexure

        res_pass = verify_aci_beam_flexure(
            mu_kip_ft=250.0,
            b_in=12.0,
            d_in=21.5,
            as_in2=3.0,
            fc_psi=4000.0,
            fy_psi=60000.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertGreaterEqual(res_pass["net_tensile_strain"], 0.005)
        self.assertEqual(res_pass["phi"], 0.90)

        res_fail_moment = verify_aci_beam_flexure(
            mu_kip_ft=350.0,
            b_in=12.0,
            d_in=21.5,
            as_in2=3.0,
            fc_psi=4000.0,
            fy_psi=60000.0,
        )
        self.assertFalse(res_fail_moment["pass"])
        self.assertGreater(res_fail_moment["dcr"], 1.0)

        res_fail_as_min = verify_aci_beam_flexure(
            mu_kip_ft=20.0,
            b_in=12.0,
            d_in=21.5,
            as_in2=0.5,
            fc_psi=4000.0,
            fy_psi=60000.0,
        )
        self.assertFalse(res_fail_as_min["pass"])

        res_nan = verify_aci_beam_flexure(
            mu_kip_ft=float("nan"),
            b_in=12.0,
            d_in=21.5,
            as_in2=3.0,
            fc_psi=4000.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aci_beam_flexure(
            mu_kip_ft=-10.0,
            b_in=12.0,
            d_in=21.5,
            as_in2=3.0,
            fc_psi=4000.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")


    def test_retaining_wall_stability(self) -> None:
        from harness.verifiers import verify_retaining_wall_stability

        res_pass = verify_retaining_wall_stability(
            stem_height_ft=10.0,
            footing_thickness_ft=1.5,
            toe_width_ft=2.0,
            stem_thickness_ft=1.0,
            heel_width_ft=5.0,
            soil_gamma_pcf=120.0,
            concrete_gamma_pcf=150.0,
            soil_friction_angle_deg=30.0,
            base_friction_coeff=0.45,
            min_fs_overturning=2.0,
            min_fs_sliding=1.5,
        )
        self.assertTrue(res_pass["pass"])
        self.assertGreaterEqual(res_pass["fs_overturning"], 2.0)
        self.assertGreaterEqual(res_pass["fs_sliding"], 1.5)
        self.assertLessEqual(res_pass["eccentricity_ft"], res_pass["kern_limit_ft"])
        self.assertAlmostEqual(res_pass["ka"], 1.0 / 3.0, places=3)

        res_fail_sliding = verify_retaining_wall_stability(
            stem_height_ft=12.0,
            footing_thickness_ft=1.0,
            toe_width_ft=1.0,
            stem_thickness_ft=1.0,
            heel_width_ft=2.0,
            base_friction_coeff=0.30,
        )
        self.assertFalse(res_fail_sliding["pass"])

        res_nan = verify_retaining_wall_stability(
            stem_height_ft=float("nan"),
            footing_thickness_ft=1.5,
            toe_width_ft=2.0,
            stem_thickness_ft=1.0,
            heel_width_ft=5.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_invalid_phi = verify_retaining_wall_stability(
            stem_height_ft=10.0,
            footing_thickness_ft=1.5,
            toe_width_ft=2.0,
            stem_thickness_ft=1.0,
            heel_width_ft=5.0,
            soil_friction_angle_deg=95.0,
        )
        self.assertFalse(res_invalid_phi["pass"])
        self.assertEqual(res_invalid_phi["reason"], "invalid_friction_angle")


    def test_aci_beam_shear(self) -> None:
        from harness.verifiers import verify_aci_beam_shear

        res_pass = verify_aci_beam_shear(
            vu_kips=40.0,
            bw_in=12.0,
            d_in=20.0,
            av_in2=0.22,
            spacing_in=8.0,
            fc_psi=4000.0,
            fyt_psi=60000.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertGreater(res_pass["phi_vn_kips"], 40.0)
        self.assertLessEqual(8.0, res_pass["s_max_in"])
        self.assertGreaterEqual(res_pass["av_s_provided"], res_pass["av_s_min"])

        res_fail_overload = verify_aci_beam_shear(
            vu_kips=85.0,
            bw_in=12.0,
            d_in=20.0,
            av_in2=0.22,
            spacing_in=8.0,
            fc_psi=4000.0,
        )
        self.assertFalse(res_fail_overload["pass"])
        self.assertGreater(res_fail_overload["dcr"], 1.0)

        res_fail_spacing = verify_aci_beam_shear(
            vu_kips=35.0,
            bw_in=12.0,
            d_in=20.0,
            av_in2=0.40,
            spacing_in=14.0,
            fc_psi=4000.0,
        )
        self.assertFalse(res_fail_spacing["pass"])

        res_nan = verify_aci_beam_shear(
            vu_kips=float("nan"),
            bw_in=12.0,
            d_in=20.0,
            av_in2=0.22,
            spacing_in=8.0,
            fc_psi=4000.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aci_beam_shear(
            vu_kips=-5.0,
            bw_in=12.0,
            d_in=20.0,
            av_in2=0.22,
            spacing_in=8.0,
            fc_psi=4000.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")


    def test_manning_pipe_flow(self) -> None:
        from harness.verifiers import verify_manning_pipe_flow

        res_pass = verify_manning_pipe_flow(
            q_design_cfs=3.0,
            diameter_in=15.0,
            slope_ft_per_ft=0.005,
            manning_n=0.013,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLessEqual(res_pass["flow_ratio"], 0.80)
        self.assertGreaterEqual(res_pass["v_full_fps"], 2.0)
        self.assertLessEqual(res_pass["v_full_fps"], 10.0)

        res_fail_cap = verify_manning_pipe_flow(
            q_design_cfs=6.0,
            diameter_in=15.0,
            slope_ft_per_ft=0.005,
        )
        self.assertFalse(res_fail_cap["pass"])
        self.assertGreater(res_fail_cap["flow_ratio"], 0.80)

        res_fail_vel = verify_manning_pipe_flow(
            q_design_cfs=1.0,
            diameter_in=15.0,
            slope_ft_per_ft=0.0005,
        )
        self.assertFalse(res_fail_vel["pass"])
        self.assertLess(res_fail_vel["v_full_fps"], 2.0)

        res_nan = verify_manning_pipe_flow(
            q_design_cfs=float("nan"),
            diameter_in=15.0,
            slope_ft_per_ft=0.005,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_manning_pipe_flow(
            q_design_cfs=-1.0,
            diameter_in=15.0,
            slope_ft_per_ft=0.005,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_flow_input")


    def test_nec_conductor_ampacity(self) -> None:
        from harness.verifiers import verify_nec_conductor_ampacity

        res_pass = verify_nec_conductor_ampacity(
            base_ampacity_a=100.0,
            continuous_load_a=50.0,
            non_continuous_load_a=10.0,
            ambient_temp_c=30.0,
            num_conductors=3,
            temp_rating_c=75.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertAlmostEqual(res_pass["dcr"], 0.725, places=3)
        self.assertEqual(res_pass["temp_factor"], 1.0)
        self.assertEqual(res_pass["bundle_factor"], 1.0)

        res_fail_derate = verify_nec_conductor_ampacity(
            base_ampacity_a=100.0,
            continuous_load_a=60.0,
            non_continuous_load_a=0.0,
            ambient_temp_c=45.0,
            num_conductors=6,
            temp_rating_c=75.0,
        )
        self.assertFalse(res_fail_derate["pass"])
        self.assertGreater(res_fail_derate["dcr"], 1.0)
        self.assertEqual(res_fail_derate["bundle_factor"], 0.80)

        res_nan = verify_nec_conductor_ampacity(
            base_ampacity_a=float("nan"),
            continuous_load_a=50.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_nec_conductor_ampacity(
            base_ampacity_a=100.0,
            continuous_load_a=-10.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")

        res_ambient_high = verify_nec_conductor_ampacity(
            base_ampacity_a=100.0,
            continuous_load_a=40.0,
            ambient_temp_c=80.0,
            temp_rating_c=75.0,
        )
        self.assertFalse(res_ambient_high["pass"])
        self.assertEqual(res_ambient_high["reason"], "ambient_exceeds_rating")


    def test_transformer_fault_current(self) -> None:
        from harness.verifiers import verify_transformer_fault_current

        res_pass = verify_transformer_fault_current(
            transformer_kva=500.0,
            secondary_voltage_v=480.0,
            impedance_pct=5.75,
            aic_rating_a=65000.0,
            three_phase=True,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertAlmostEqual(res_pass["fla_a"], 601.41, places=1)
        self.assertAlmostEqual(res_pass["total_fault_current_a"], 10459.25, places=1)

        res_fail = verify_transformer_fault_current(
            transformer_kva=500.0,
            secondary_voltage_v=480.0,
            impedance_pct=5.75,
            aic_rating_a=10000.0,
            three_phase=True,
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr"], 1.0)

        res_motor = verify_transformer_fault_current(
            transformer_kva=500.0,
            secondary_voltage_v=480.0,
            impedance_pct=5.75,
            aic_rating_a=65000.0,
            motor_contribution_multiplier=4.0,
        )
        self.assertTrue(res_motor["pass"])
        self.assertGreater(res_motor["total_fault_current_a"], res_pass["total_fault_current_a"])
        self.assertAlmostEqual(res_motor["motor_contribution_a"], 4.0 * res_pass["fla_a"], places=1)

        res_nan = verify_transformer_fault_current(
            transformer_kva=float("nan"),
            secondary_voltage_v=480.0,
            impedance_pct=5.75,
            aic_rating_a=65000.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_transformer_fault_current(
            transformer_kva=-500.0,
            secondary_voltage_v=480.0,
            impedance_pct=5.75,
            aic_rating_a=65000.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_rating_or_voltage")


    def test_hydronic_pipe_sizing(self) -> None:
        from harness.verifiers import verify_hydronic_pipe_sizing

        res_pass = verify_hydronic_pipe_sizing(
            flow_gpm=35.0,
            inner_diameter_in=2.067,
            pipe_length_ft=150.0,
            c_factor=130.0,
            max_velocity_fps=8.0,
            min_velocity_fps=2.0,
            max_head_loss_per_100ft=4.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLessEqual(res_pass["velocity_fps"], 8.0)
        self.assertGreaterEqual(res_pass["velocity_fps"], 2.0)
        self.assertLessEqual(res_pass["head_loss_per_100ft"], 4.0)
        self.assertTrue(res_pass["velocity_ok"])

        res_fail_vel = verify_hydronic_pipe_sizing(
            flow_gpm=120.0,
            inner_diameter_in=2.067,
        )
        self.assertFalse(res_fail_vel["pass"])
        self.assertGreater(res_fail_vel["velocity_dcr"], 1.0)

        res_fail_min_vel = verify_hydronic_pipe_sizing(
            flow_gpm=5.0,
            inner_diameter_in=2.067,
        )
        self.assertFalse(res_fail_min_vel["pass"])
        self.assertLess(res_fail_min_vel["velocity_fps"], 2.0)

        res_nan = verify_hydronic_pipe_sizing(
            flow_gpm=float("nan"),
            inner_diameter_in=2.067,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_hydronic_pipe_sizing(
            flow_gpm=-20.0,
            inner_diameter_in=2.067,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_geometry_or_flow")

        res_inv_limits = verify_hydronic_pipe_sizing(
            flow_gpm=35.0,
            inner_diameter_in=2.067,
            min_velocity_fps=10.0,
            max_velocity_fps=5.0,
        )
        self.assertFalse(res_inv_limits["pass"])
        self.assertEqual(res_inv_limits["reason"], "invalid_velocity_limits")


    def test_asce7_seismic_base_shear(self) -> None:
        from harness.verifiers import verify_asce7_seismic_base_shear

        res_pass = verify_asce7_seismic_base_shear(
            seismic_weight_kips=2500.0,
            s_ds=1.0,
            s_d1=0.6,
            fundamental_period_s=0.8,
            r_factor=8.0,
            importance_factor=1.0,
            tl_period_s=8.0,
            s1=0.5,
            v_capacity_kips=250.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertAlmostEqual(res_pass["base_shear_kips"], 234.38, places=1)
        self.assertAlmostEqual(res_pass["cs"], 0.09375, places=4)

        res_fail = verify_asce7_seismic_base_shear(
            seismic_weight_kips=2500.0,
            s_ds=1.0,
            s_d1=0.6,
            fundamental_period_s=0.8,
            r_factor=8.0,
            v_capacity_kips=200.0,
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr"], 1.0)

        res_long_period = verify_asce7_seismic_base_shear(
            seismic_weight_kips=3000.0,
            s_ds=0.8,
            s_d1=0.4,
            fundamental_period_s=9.0,
            r_factor=6.0,
            tl_period_s=8.0,
            v_capacity_kips=150.0,
        )
        self.assertTrue(res_long_period["pass"])
        self.assertGreaterEqual(res_long_period["cs"], res_long_period["cs_min"])

        res_nan = verify_asce7_seismic_base_shear(
            seismic_weight_kips=float("nan"),
            s_ds=1.0,
            s_d1=0.6,
            fundamental_period_s=0.8,
            r_factor=8.0,
            v_capacity_kips=250.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_asce7_seismic_base_shear(
            seismic_weight_kips=-2500.0,
            s_ds=1.0,
            s_d1=0.6,
            fundamental_period_s=0.8,
            r_factor=8.0,
            v_capacity_kips=250.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_geometry_or_hazard")


    def test_ashrae_ventilation_rate(self) -> None:
        from harness.verifiers import verify_ashrae_ventilation_rate

        res_pass = verify_ashrae_ventilation_rate(
            floor_area_sqft=1000.0,
            occupant_count=20.0,
            rp_cfm_per_person=5.0,
            ra_cfm_per_sqft=0.06,
            supplied_oa_cfm=200.0,
            ez_effectiveness=1.0,
            supply_air_cfm=1000.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["v_bz_cfm"], 160.0)
        self.assertEqual(res_pass["v_oz_cfm"], 160.0)
        self.assertAlmostEqual(res_pass["dcr"], 0.8, places=3)
        self.assertAlmostEqual(res_pass["z_p"], 0.16, places=3)

        res_fail_oa = verify_ashrae_ventilation_rate(
            floor_area_sqft=1000.0,
            occupant_count=20.0,
            rp_cfm_per_person=5.0,
            ra_cfm_per_sqft=0.06,
            supplied_oa_cfm=120.0,
        )
        self.assertFalse(res_fail_oa["pass"])
        self.assertGreater(res_fail_oa["dcr"], 1.0)

        res_fail_heating = verify_ashrae_ventilation_rate(
            floor_area_sqft=1000.0,
            occupant_count=20.0,
            rp_cfm_per_person=5.0,
            ra_cfm_per_sqft=0.06,
            supplied_oa_cfm=180.0,
            ez_effectiveness=0.8,
        )
        self.assertFalse(res_fail_heating["pass"])
        self.assertEqual(res_fail_heating["v_oz_cfm"], 200.0)
        self.assertGreater(res_fail_heating["dcr"], 1.0)

        res_fail_zp = verify_ashrae_ventilation_rate(
            floor_area_sqft=1000.0,
            occupant_count=20.0,
            rp_cfm_per_person=5.0,
            ra_cfm_per_sqft=0.06,
            supplied_oa_cfm=180.0,
            supply_air_cfm=150.0,
        )
        self.assertFalse(res_fail_zp["pass"])
        self.assertEqual(res_fail_zp["reason"], "supplied_oa_exceeds_total_supply_air")

        res_nan = verify_ashrae_ventilation_rate(
            floor_area_sqft=float("nan"),
            occupant_count=20.0,
            rp_cfm_per_person=5.0,
            ra_cfm_per_sqft=0.06,
            supplied_oa_cfm=200.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_ashrae_ventilation_rate(
            floor_area_sqft=1000.0,
            occupant_count=-5.0,
            rp_cfm_per_person=5.0,
            ra_cfm_per_sqft=0.06,
            supplied_oa_cfm=200.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_occupancy")


    def test_aisc_bolted_connection(self) -> None:
        from harness.verifiers import verify_aisc_bolted_connection

        res_pass = verify_aisc_bolted_connection(
            vu_kips=50.0,
            num_bolts=4,
            bolt_diameter_in=0.75,
            f_nv_ksi=54.0,
            plate_thickness_in=0.50,
            fu_ksi=58.0,
            clear_distance_lc_in=1.5,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertEqual(res_pass["governing_limit_state"], "bolt_shear")
        self.assertAlmostEqual(res_pass["phi_rnv_kips"], 71.57, places=1)

        res_fail_shear = verify_aisc_bolted_connection(
            vu_kips=85.0,
            num_bolts=4,
            bolt_diameter_in=0.75,
            f_nv_ksi=54.0,
            plate_thickness_in=0.50,
            fu_ksi=58.0,
            clear_distance_lc_in=1.5,
        )
        self.assertFalse(res_fail_shear["pass"])
        self.assertGreater(res_fail_shear["dcr"], 1.0)

        res_bearing_gov = verify_aisc_bolted_connection(
            vu_kips=30.0,
            num_bolts=4,
            bolt_diameter_in=0.75,
            f_nv_ksi=68.0,
            plate_thickness_in=0.25,
            fu_ksi=58.0,
            clear_distance_lc_in=0.50,
        )
        self.assertFalse(res_bearing_gov["pass"])
        self.assertEqual(res_bearing_gov["governing_limit_state"], "plate_bearing")

        res_nan = verify_aisc_bolted_connection(
            vu_kips=float("nan"),
            num_bolts=4,
            bolt_diameter_in=0.75,
            f_nv_ksi=54.0,
            plate_thickness_in=0.50,
            fu_ksi=58.0,
            clear_distance_lc_in=1.5,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aisc_bolted_connection(
            vu_kips=-10.0,
            num_bolts=4,
            bolt_diameter_in=0.75,
            f_nv_ksi=54.0,
            plate_thickness_in=0.50,
            fu_ksi=58.0,
            clear_distance_lc_in=1.5,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")

        res_inv_bolts = verify_aisc_bolted_connection(
            vu_kips=50.0,
            num_bolts=0,
            bolt_diameter_in=0.75,
            f_nv_ksi=54.0,
            plate_thickness_in=0.50,
            fu_ksi=58.0,
            clear_distance_lc_in=1.5,
        )
        self.assertFalse(res_inv_bolts["pass"])
        self.assertEqual(res_inv_bolts["reason"], "invalid_bolt_count_or_shear_planes")


    def test_pump_npsh(self) -> None:
        from harness.verifiers import verify_pump_npsh

        res_pass = verify_pump_npsh(
            npsh_required_ft=12.0,
            static_suction_head_ft=4.0,
            friction_head_loss_ft=2.0,
            atmospheric_pressure_psia=14.696,
            vapor_pressure_psia=0.363,
            specific_gravity=1.0,
            min_margin_ratio=1.20,
            min_margin_ft=2.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertGreater(res_pass["npsh_available_ft"], 12.0)
        self.assertGreaterEqual(res_pass["margin_ratio"], 1.20)
        self.assertGreaterEqual(res_pass["margin_ft"], 2.0)

        res_fail_margin = verify_pump_npsh(
            npsh_required_ft=10.0,
            static_suction_head_ft=-18.0,
            friction_head_loss_ft=4.0,
            atmospheric_pressure_psia=14.696,
            vapor_pressure_psia=0.363,
            specific_gravity=1.0,
        )
        self.assertFalse(res_fail_margin["pass"])
        self.assertGreater(res_fail_margin["dcr"], 1.0)

        res_fail_neg_npsha = verify_pump_npsh(
            npsh_required_ft=8.0,
            static_suction_head_ft=-35.0,
            friction_head_loss_ft=2.0,
        )
        self.assertFalse(res_fail_neg_npsha["pass"])
        self.assertEqual(res_fail_neg_npsha["reason"], "zero_or_negative_npsh_available")

        res_nan = verify_pump_npsh(
            npsh_required_ft=float("nan"),
            static_suction_head_ft=4.0,
            friction_head_loss_ft=2.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_vp_err = verify_pump_npsh(
            npsh_required_ft=10.0,
            static_suction_head_ft=5.0,
            friction_head_loss_ft=1.0,
            atmospheric_pressure_psia=14.7,
            vapor_pressure_psia=15.0,
        )
        self.assertFalse(res_vp_err["pass"])
        self.assertEqual(res_vp_err["reason"], "vapor_pressure_exceeds_atmospheric")


    def test_asce7_wind_pressure(self) -> None:
        from harness.verifiers import verify_asce7_wind_pressure

        res_pass = verify_asce7_wind_pressure(
            basic_wind_speed_mph=115.0,
            height_z_ft=30.0,
            mean_roof_height_ft=30.0,
            exposure_category="C",
            p_capacity_psf=35.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertAlmostEqual(res_pass["kz"], 0.9823, places=2)
        self.assertAlmostEqual(res_pass["p_design_psf"], 24.30, places=1)

        res_fail_capacity = verify_asce7_wind_pressure(
            basic_wind_speed_mph=160.0,
            height_z_ft=30.0,
            mean_roof_height_ft=30.0,
            exposure_category="C",
            p_capacity_psf=30.0,
        )
        self.assertFalse(res_fail_capacity["pass"])
        self.assertGreater(res_fail_capacity["dcr"], 1.0)

        res_min_pressure = verify_asce7_wind_pressure(
            basic_wind_speed_mph=50.0,
            height_z_ft=15.0,
            mean_roof_height_ft=15.0,
            exposure_category="B",
            p_capacity_psf=20.0,
            min_pressure_psf=16.0,
        )
        self.assertTrue(res_min_pressure["pass"])
        self.assertEqual(res_min_pressure["p_design_psf"], 16.0)

        res_height_err = verify_asce7_wind_pressure(
            basic_wind_speed_mph=115.0,
            height_z_ft=40.0,
            mean_roof_height_ft=30.0,
        )
        self.assertFalse(res_height_err["pass"])
        self.assertEqual(res_height_err["reason"], "height_exceeds_roof_height")

        res_exposure_err = verify_asce7_wind_pressure(
            basic_wind_speed_mph=115.0,
            height_z_ft=30.0,
            mean_roof_height_ft=30.0,
            exposure_category="A",
        )
        self.assertFalse(res_exposure_err["pass"])
        self.assertEqual(res_exposure_err["reason"], "invalid_exposure_category")

        res_nan = verify_asce7_wind_pressure(
            basic_wind_speed_mph=float("nan"),
            height_z_ft=30.0,
            mean_roof_height_ft=30.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_asce7_wind_pressure(
            basic_wind_speed_mph=-115.0,
            height_z_ft=30.0,
            mean_roof_height_ft=30.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_geometry_or_velocity")


    def test_nec_solar_busbar_interconnection(self) -> None:
        from harness.verifiers import verify_nec_solar_busbar_interconnection

        res_pass = verify_nec_solar_busbar_interconnection(
            busbar_rating_a=200.0,
            main_breaker_a=200.0,
            inverter_rated_ac_current_a=32.0,
            opposite_end=True,
            allowance_pct=120.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertAlmostEqual(res_pass["dcr"], 1.0, places=3)
        self.assertEqual(res_pass["pv_breaker_a"], 40.0)
        self.assertEqual(res_pass["max_allowed_supply_a"], 240.0)
        self.assertEqual(res_pass["max_pv_breaker_a"], 40.0)

        res_fail_overfeed = verify_nec_solar_busbar_interconnection(
            busbar_rating_a=200.0,
            main_breaker_a=200.0,
            inverter_rated_ac_current_a=48.0,
            opposite_end=True,
        )
        self.assertFalse(res_fail_overfeed["pass"])
        self.assertGreater(res_fail_overfeed["dcr"], 1.0)

        res_fail_non_opposite = verify_nec_solar_busbar_interconnection(
            busbar_rating_a=200.0,
            main_breaker_a=200.0,
            inverter_rated_ac_current_a=32.0,
            opposite_end=False,
        )
        self.assertFalse(res_fail_non_opposite["pass"])
        self.assertGreater(res_fail_non_opposite["dcr"], 1.0)
        self.assertEqual(res_fail_non_opposite["max_allowed_supply_a"], 200.0)

        res_undersized_breaker = verify_nec_solar_busbar_interconnection(
            busbar_rating_a=200.0,
            main_breaker_a=150.0,
            inverter_rated_ac_current_a=40.0,
            pv_breaker_a=40.0,
        )
        self.assertFalse(res_undersized_breaker["pass"])
        self.assertEqual(res_undersized_breaker["reason"], "inverter_ocpd_undersized")

        res_nan = verify_nec_solar_busbar_interconnection(
            busbar_rating_a=float("nan"),
            main_breaker_a=200.0,
            inverter_rated_ac_current_a=32.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_nec_solar_busbar_interconnection(
            busbar_rating_a=-200.0,
            main_breaker_a=200.0,
            inverter_rated_ac_current_a=32.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_rating_or_current")


    def test_aisc_fillet_weld(self) -> None:
        from harness.verifiers import verify_aisc_fillet_weld

        res_pass = verify_aisc_fillet_weld(
            pu_kips=40.0,
            leg_size_in=0.3125,
            length_in=8.0,
            f_exx_ksi=70.0,
            base_metal_fu_ksi=65.0,
            base_metal_t_in=0.50,
            load_angle_deg=0.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertEqual(res_pass["governing_limit_state"], "weld_shear")
        self.assertAlmostEqual(res_pass["phi_rn_weld_kips"], 55.68, places=1)
        self.assertEqual(res_pass["beta"], 1.0)
        self.assertEqual(res_pass["k_ds"], 1.0)

        res_fail = verify_aisc_fillet_weld(
            pu_kips=70.0,
            leg_size_in=0.3125,
            length_in=8.0,
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr"], 1.0)

        res_transverse = verify_aisc_fillet_weld(
            pu_kips=70.0,
            leg_size_in=0.3125,
            length_in=8.0,
            load_angle_deg=90.0,
        )
        self.assertTrue(res_transverse["pass"])
        self.assertEqual(res_transverse["k_ds"], 1.5)
        self.assertAlmostEqual(res_transverse["phi_rn_weld_kips"], 83.53, places=1)

        res_base_gov = verify_aisc_fillet_weld(
            pu_kips=30.0,
            leg_size_in=0.3125,
            length_in=8.0,
            base_metal_t_in=0.15,
        )
        self.assertEqual(res_base_gov["governing_limit_state"], "base_metal_shear")
        self.assertAlmostEqual(res_base_gov["phi_rn_base_kips"], 35.1, places=1)

        res_long_weld = verify_aisc_fillet_weld(
            pu_kips=100.0,
            leg_size_in=0.25,
            length_in=35.0,
        )
        self.assertEqual(res_long_weld["beta"], 0.92)

        res_nan = verify_aisc_fillet_weld(
            pu_kips=float("nan"),
            leg_size_in=0.3125,
            length_in=8.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aisc_fillet_weld(
            pu_kips=-10.0,
            leg_size_in=0.3125,
            length_in=8.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")

        res_inv_angle = verify_aisc_fillet_weld(
            pu_kips=40.0,
            leg_size_in=0.3125,
            length_in=8.0,
            load_angle_deg=105.0,
        )
        self.assertFalse(res_inv_angle["pass"])
        self.assertEqual(res_inv_angle["reason"], "invalid_load_angle")


    def test_nec_conduit_fill(self) -> None:
        from harness.verifiers import verify_nec_conduit_fill

        res_pass = verify_nec_conduit_fill(
            conduit_inner_diameter_in=1.049,
            conductor_diameters_in=[0.30, 0.30, 0.30],
            conduit_length_in=120.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["allowable_fill_pct"], 40.0)
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertEqual(res_pass["conductor_count"], 3)

        res_fail = verify_nec_conduit_fill(
            conduit_inner_diameter_in=1.049,
            conductor_diameters_in=[0.40, 0.40, 0.40, 0.40],
            conduit_length_in=120.0,
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr"], 1.0)
        self.assertEqual(res_fail["allowable_fill_pct"], 40.0)

        res_nipple = verify_nec_conduit_fill(
            conduit_inner_diameter_in=1.049,
            conductor_diameters_in=[0.40, 0.40, 0.40, 0.40],
            conduit_length_in=20.0,
        )
        self.assertTrue(res_nipple["pass"])
        self.assertEqual(res_nipple["allowable_fill_pct"], 60.0)
        self.assertLess(res_nipple["dcr"], 1.0)

        res_two = verify_nec_conduit_fill(
            conduit_inner_diameter_in=1.049,
            conductor_diameters_in=[0.45, 0.45],
            conduit_length_in=60.0,
        )
        self.assertFalse(res_two["pass"])
        self.assertEqual(res_two["allowable_fill_pct"], 31.0)

        res_one = verify_nec_conduit_fill(
            conduit_inner_diameter_in=1.049,
            conductor_diameters_in=[0.60],
            conduit_length_in=60.0,
        )
        self.assertTrue(res_one["pass"])
        self.assertEqual(res_one["allowable_fill_pct"], 53.0)

        res_exceed = verify_nec_conduit_fill(
            conduit_inner_diameter_in=1.0,
            conductor_diameters_in=[1.25],
        )
        self.assertFalse(res_exceed["pass"])
        self.assertEqual(res_exceed["reason"], "conductor_exceeds_conduit_diameter")

        res_empty = verify_nec_conduit_fill(
            conduit_inner_diameter_in=1.049,
            conductor_diameters_in=[],
        )
        self.assertFalse(res_empty["pass"])
        self.assertEqual(res_empty["reason"], "empty_conductor_list")

        res_nan = verify_nec_conduit_fill(
            conduit_inner_diameter_in=float("nan"),
            conductor_diameters_in=[0.30],
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_nec_conduit_fill(
            conduit_inner_diameter_in=-1.0,
            conductor_diameters_in=[0.30],
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_conduit_diameter")


    def test_aisc_tension_member(self) -> None:
        from harness.verifiers import verify_aisc_tension_member

        res_pass = verify_aisc_tension_member(
            pu_kips=150.0,
            ag_in2=6.0,
            an_in2=5.0,
            u_shear_lag=0.85,
            length_in=180.0,
            rx_in=2.5,
            ry_in=1.5,
            fy_ksi=50.0,
            fu_ksi=65.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertEqual(res_pass["governing_limit_state"], "net_rupture")
        self.assertAlmostEqual(res_pass["phi_pny_kips"], 270.0, places=1)
        self.assertAlmostEqual(res_pass["phi_pnr_kips"], 207.19, places=1)
        self.assertEqual(res_pass["slenderness_ratio"], 120.0)

        res_yield = verify_aisc_tension_member(
            pu_kips=180.0,
            ag_in2=4.5,
            an_in2=4.2,
            u_shear_lag=1.0,
            length_in=120.0,
            rx_in=2.0,
            ry_in=1.2,
            fy_ksi=50.0,
            fu_ksi=65.0,
        )
        self.assertTrue(res_yield["pass"])
        self.assertEqual(res_yield["governing_limit_state"], "gross_yielding")
        self.assertAlmostEqual(res_yield["phi_pny_kips"], 202.5, places=1)

        res_fail_dcr = verify_aisc_tension_member(
            pu_kips=220.0,
            ag_in2=6.0,
            an_in2=5.0,
            u_shear_lag=0.85,
            length_in=180.0,
            rx_in=2.5,
            ry_in=1.5,
        )
        self.assertFalse(res_fail_dcr["pass"])
        self.assertGreater(res_fail_dcr["dcr"], 1.0)

        res_fail_slender = verify_aisc_tension_member(
            pu_kips=50.0,
            ag_in2=6.0,
            an_in2=5.0,
            u_shear_lag=0.85,
            length_in=480.0,
            rx_in=2.5,
            ry_in=1.0,
        )
        self.assertFalse(res_fail_slender["pass"])
        self.assertGreater(res_fail_slender["slenderness_ratio"], 300.0)

        res_area_err = verify_aisc_tension_member(
            pu_kips=50.0,
            ag_in2=4.0,
            an_in2=5.0,
            u_shear_lag=0.85,
            length_in=120.0,
            rx_in=2.0,
            ry_in=1.0,
        )
        self.assertFalse(res_area_err["pass"])
        self.assertEqual(res_area_err["reason"], "net_area_exceeds_gross_area")

        res_u_err = verify_aisc_tension_member(
            pu_kips=50.0,
            ag_in2=6.0,
            an_in2=5.0,
            u_shear_lag=1.2,
            length_in=120.0,
            rx_in=2.0,
            ry_in=1.0,
        )
        self.assertFalse(res_u_err["pass"])
        self.assertEqual(res_u_err["reason"], "invalid_shear_lag_factor")

        res_nan = verify_aisc_tension_member(
            pu_kips=float("nan"),
            ag_in2=6.0,
            an_in2=5.0,
            u_shear_lag=0.85,
            length_in=120.0,
            rx_in=2.0,
            ry_in=1.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aisc_tension_member(
            pu_kips=-10.0,
            ag_in2=6.0,
            an_in2=5.0,
            u_shear_lag=0.85,
            length_in=120.0,
            rx_in=2.0,
            ry_in=1.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")


    def test_clay_consolidation_settlement(self) -> None:
        from harness.verifiers import verify_clay_consolidation_settlement

        res_pass_oc = verify_clay_consolidation_settlement(
            layer_thickness_ft=10.0,
            initial_void_ratio=0.80,
            compression_index_cc=0.30,
            recompression_index_cr=0.05,
            initial_effective_stress_psf=2000.0,
            preconsolidation_stress_psf=3000.0,
            stress_increase_psf=2000.0,
            allowable_settlement_in=4.0,
        )
        self.assertTrue(res_pass_oc["pass"])
        self.assertLess(res_pass_oc["dcr"], 1.0)
        self.assertEqual(res_pass_oc["consolidation_state"], "overconsolidated_virgin")
        self.assertAlmostEqual(res_pass_oc["ocr"], 1.5, places=2)
        self.assertAlmostEqual(res_pass_oc["settlement_in"], 3.0858, places=3)

        res_elastic = verify_clay_consolidation_settlement(
            layer_thickness_ft=10.0,
            initial_void_ratio=0.80,
            compression_index_cc=0.30,
            recompression_index_cr=0.05,
            initial_effective_stress_psf=2000.0,
            preconsolidation_stress_psf=3000.0,
            stress_increase_psf=500.0,
            allowable_settlement_in=1.0,
        )
        self.assertTrue(res_elastic["pass"])
        self.assertEqual(res_elastic["consolidation_state"], "overconsolidated_elastic")
        self.assertLess(res_elastic["settlement_in"], 0.50)

        res_fail_nc = verify_clay_consolidation_settlement(
            layer_thickness_ft=15.0,
            initial_void_ratio=0.90,
            compression_index_cc=0.40,
            recompression_index_cr=0.06,
            initial_effective_stress_psf=1500.0,
            preconsolidation_stress_psf=1500.0,
            stress_increase_psf=1800.0,
            allowable_settlement_in=1.5,
        )
        self.assertFalse(res_fail_nc["pass"])
        self.assertEqual(res_fail_nc["consolidation_state"], "normally_consolidated")
        self.assertGreater(res_fail_nc["dcr"], 1.0)

        res_cr_err = verify_clay_consolidation_settlement(
            layer_thickness_ft=10.0,
            initial_void_ratio=0.80,
            compression_index_cc=0.10,
            recompression_index_cr=0.25,
            initial_effective_stress_psf=2000.0,
            preconsolidation_stress_psf=3000.0,
            stress_increase_psf=1000.0,
        )
        self.assertFalse(res_cr_err["pass"])
        self.assertEqual(res_cr_err["reason"], "recompression_index_exceeds_compression_index")

        res_nan = verify_clay_consolidation_settlement(
            layer_thickness_ft=float("nan"),
            initial_void_ratio=0.80,
            compression_index_cc=0.30,
            recompression_index_cr=0.05,
            initial_effective_stress_psf=2000.0,
            preconsolidation_stress_psf=3000.0,
            stress_increase_psf=1000.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_clay_consolidation_settlement(
            layer_thickness_ft=10.0,
            initial_void_ratio=0.80,
            compression_index_cc=0.30,
            recompression_index_cr=0.05,
            initial_effective_stress_psf=2000.0,
            preconsolidation_stress_psf=3000.0,
            stress_increase_psf=-500.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_stress_increase")


    def test_nec_motor_branch_circuit(self) -> None:
        from harness.verifiers import verify_nec_motor_branch_circuit

        res_pass = verify_nec_motor_branch_circuit(
            motor_fla_a=14.0,
            conductor_ampacity_a=25.0,
            ocpd_rating_a=35.0,
            ocpd_type="inverse_time_breaker",
            overload_rating_a=16.0,
            service_factor=1.15,
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["min_conductor_ampacity_a"], 17.5)
        self.assertEqual(res_pass["max_allowed_ocpd_a"], 35.0)
        self.assertEqual(res_pass["max_overload_a"], 17.5)
        self.assertLessEqual(res_pass["conductor_dcr"], 1.0)
        self.assertLessEqual(res_pass["ocpd_dcr"], 1.0)
        self.assertLessEqual(res_pass["overload_dcr"], 1.0)

        res_next_size = verify_nec_motor_branch_circuit(
            motor_fla_a=28.0,
            conductor_ampacity_a=40.0,
            ocpd_rating_a=50.0,
            ocpd_type="dual_element_fuse",
            allow_next_standard_ocpd=True,
        )
        self.assertTrue(res_next_size["pass"])
        self.assertEqual(res_next_size["max_allowed_ocpd_a"], 50.0)
        self.assertEqual(res_next_size["ocpd_dcr"], 1.0)

        res_fail_conductor = verify_nec_motor_branch_circuit(
            motor_fla_a=28.0,
            conductor_ampacity_a=30.0,
            ocpd_rating_a=70.0,
            ocpd_type="inverse_time_breaker",
        )
        self.assertFalse(res_fail_conductor["pass"])
        self.assertFalse(res_fail_conductor["conductor_ok"])
        self.assertGreater(res_fail_conductor["conductor_dcr"], 1.0)

        res_fail_ocpd = verify_nec_motor_branch_circuit(
            motor_fla_a=14.0,
            conductor_ampacity_a=25.0,
            ocpd_rating_a=50.0,
            ocpd_type="inverse_time_breaker",
        )
        self.assertFalse(res_fail_ocpd["pass"])
        self.assertFalse(res_fail_ocpd["ocpd_ok"])
        self.assertGreater(res_fail_ocpd["ocpd_dcr"], 1.0)

        res_fail_overload = verify_nec_motor_branch_circuit(
            motor_fla_a=14.0,
            conductor_ampacity_a=25.0,
            ocpd_rating_a=35.0,
            overload_rating_a=19.0,
            service_factor=1.15,
        )
        self.assertFalse(res_fail_overload["pass"])
        self.assertFalse(res_fail_overload["overload_ok"])
        self.assertGreater(res_fail_overload["overload_dcr"], 1.0)

        res_nan = verify_nec_motor_branch_circuit(
            motor_fla_a=float("nan"),
            conductor_ampacity_a=25.0,
            ocpd_rating_a=35.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_nec_motor_branch_circuit(
            motor_fla_a=-14.0,
            conductor_ampacity_a=25.0,
            ocpd_rating_a=35.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_current_or_ampacity")

        res_inv_type = verify_nec_motor_branch_circuit(
            motor_fla_a=14.0,
            conductor_ampacity_a=25.0,
            ocpd_rating_a=35.0,
            ocpd_type="magnetic_only",
        )
        self.assertFalse(res_inv_type["pass"])
        self.assertEqual(res_inv_type["reason"], "invalid_ocpd_type")


    def test_stormwater_rational_runoff(self) -> None:
        from harness.verifiers import verify_stormwater_rational_runoff

        res_pass = verify_stormwater_rational_runoff(
            drainage_areas_acres=[2.5, 1.5, 1.0],
            runoff_coefficients=[0.90, 0.70, 0.30],
            rainfall_intensity_in_hr=4.5,
            capacity_cfs=20.0,
            frequency_factor_cf=1.1,
            time_of_concentration_min=10.0,
            min_time_of_concentration_min=5.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertAlmostEqual(res_pass["composite_c"], 0.72, places=3)
        self.assertAlmostEqual(res_pass["design_c"], 0.792, places=3)
        self.assertAlmostEqual(res_pass["q_peak_cfs"], 17.82, places=2)
        self.assertAlmostEqual(res_pass["dcr"], 0.891, places=3)

        res_fail = verify_stormwater_rational_runoff(
            drainage_areas_acres=[2.5, 1.5, 1.0],
            runoff_coefficients=[0.90, 0.70, 0.30],
            rainfall_intensity_in_hr=4.5,
            capacity_cfs=15.0,
            frequency_factor_cf=1.1,
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr"], 1.0)

        res_tc_err = verify_stormwater_rational_runoff(
            drainage_areas_acres=[1.0],
            runoff_coefficients=[0.5],
            rainfall_intensity_in_hr=3.0,
            capacity_cfs=5.0,
            time_of_concentration_min=3.0,
            min_time_of_concentration_min=5.0,
        )
        self.assertFalse(res_tc_err["pass"])
        self.assertEqual(res_tc_err["reason"], "tc_below_minimum")

        res_mismatch = verify_stormwater_rational_runoff(
            drainage_areas_acres=[1.0, 2.0],
            runoff_coefficients=[0.5],
            rainfall_intensity_in_hr=3.0,
            capacity_cfs=5.0,
        )
        self.assertFalse(res_mismatch["pass"])
        self.assertEqual(res_mismatch["reason"], "length_mismatch")

        res_empty = verify_stormwater_rational_runoff(
            drainage_areas_acres=[],
            runoff_coefficients=[],
            rainfall_intensity_in_hr=3.0,
            capacity_cfs=5.0,
        )
        self.assertFalse(res_empty["pass"])
        self.assertEqual(res_empty["reason"], "empty_subcatchment_list")

        res_nan = verify_stormwater_rational_runoff(
            drainage_areas_acres=[float("nan")],
            runoff_coefficients=[0.5],
            rainfall_intensity_in_hr=3.0,
            capacity_cfs=5.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_coeff_err = verify_stormwater_rational_runoff(
            drainage_areas_acres=[1.0],
            runoff_coefficients=[1.2],
            rainfall_intensity_in_hr=3.0,
            capacity_cfs=5.0,
        )
        self.assertFalse(res_coeff_err["pass"])
        self.assertEqual(res_coeff_err["reason"], "invalid_runoff_coefficient")


    def test_aci_punching_shear(self) -> None:
        from harness.verifiers import verify_aci_punching_shear

        res_pass = verify_aci_punching_shear(
            vu_kips=120.0,
            c1_in=16.0,
            c2_in=16.0,
            d_in=8.0,
            fc_psi=4000.0,
            column_location="interior",
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertEqual(res_pass["b0_in"], 96.0)
        self.assertEqual(res_pass["lambda_s"], 1.0)
        self.assertEqual(res_pass["governing_equation"], "4_sqrt_fc")
        self.assertAlmostEqual(res_pass["phi_vn_kips"], 145.72, places=1)

        res_beta_gov = verify_aci_punching_shear(
            vu_kips=150.0,
            c1_in=12.0,
            c2_in=48.0,
            d_in=8.0,
            fc_psi=4000.0,
            column_location="interior",
        )
        self.assertEqual(res_beta_gov["governing_equation"], "aspect_ratio_beta")
        self.assertEqual(res_beta_gov["beta"], 4.0)

        res_thick_slab = verify_aci_punching_shear(
            vu_kips=200.0,
            c1_in=18.0,
            c2_in=18.0,
            d_in=20.0,
            fc_psi=4000.0,
            column_location="interior",
        )
        self.assertLess(res_thick_slab["lambda_s"], 1.0)
        self.assertAlmostEqual(res_thick_slab["lambda_s"], (2.0 / 3.0) ** 0.5, places=3)

        res_fail_overload = verify_aci_punching_shear(
            vu_kips=200.0,
            c1_in=16.0,
            c2_in=16.0,
            d_in=8.0,
            fc_psi=4000.0,
            column_location="interior",
        )
        self.assertFalse(res_fail_overload["pass"])
        self.assertGreater(res_fail_overload["dcr"], 1.0)

        res_nan = verify_aci_punching_shear(
            vu_kips=float("nan"),
            c1_in=16.0,
            c2_in=16.0,
            d_in=8.0,
            fc_psi=4000.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aci_punching_shear(
            vu_kips=-50.0,
            c1_in=16.0,
            c2_in=16.0,
            d_in=8.0,
            fc_psi=4000.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")

        res_inv_loc = verify_aci_punching_shear(
            vu_kips=100.0,
            c1_in=16.0,
            c2_in=16.0,
            d_in=8.0,
            fc_psi=4000.0,
            column_location="cantilever",
        )
        self.assertFalse(res_inv_loc["pass"])
        self.assertEqual(res_inv_loc["reason"], "invalid_column_location")


    def test_hvac_cooling_load_capacity(self) -> None:
        from harness.verifiers import verify_hvac_cooling_load_capacity

        res_pass = verify_hvac_cooling_load_capacity(
            sensible_load_btuh=36000.0,
            latent_load_btuh=12000.0,
            room_temp_db_f=75.0,
            supply_temp_db_f=55.0,
            design_airflow_cfm=1800.0,
            coil_capacity_tons=4.5,
            elevation_ft=0.0,
            room_volume_cuft=12000.0,
            min_ach=6.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["q_total_btuh"], 48000.0)
        self.assertEqual(res_pass["tons_required"], 4.0)
        self.assertEqual(res_pass["shr"], 0.75)
        self.assertAlmostEqual(res_pass["required_cfm"], 1666.67, places=1)
        self.assertLess(res_pass["airflow_dcr"], 1.0)
        self.assertLess(res_pass["capacity_dcr"], 1.0)
        self.assertEqual(res_pass["ach"], 9.0)

        # High elevation altitude derating triggers airflow deficit
        res_altitude = verify_hvac_cooling_load_capacity(
            sensible_load_btuh=36000.0,
            latent_load_btuh=12000.0,
            room_temp_db_f=75.0,
            supply_temp_db_f=55.0,
            design_airflow_cfm=1800.0,
            coil_capacity_tons=4.5,
            elevation_ft=5280.0,
        )
        self.assertFalse(res_altitude["pass"])
        self.assertGreater(res_altitude["airflow_dcr"], 1.0)
        self.assertLess(res_altitude["density_ratio"], 0.85)

        # Undersized coil capacity failure
        res_undersized_coil = verify_hvac_cooling_load_capacity(
            sensible_load_btuh=36000.0,
            latent_load_btuh=12000.0,
            room_temp_db_f=75.0,
            supply_temp_db_f=55.0,
            design_airflow_cfm=2000.0,
            coil_capacity_tons=3.5,
        )
        self.assertFalse(res_undersized_coil["pass"])
        self.assertGreater(res_undersized_coil["capacity_dcr"], 1.0)

        # Room temperature equal to or less than supply temperature
        res_inv_temp = verify_hvac_cooling_load_capacity(
            sensible_load_btuh=36000.0,
            latent_load_btuh=12000.0,
            room_temp_db_f=55.0,
            supply_temp_db_f=55.0,
            design_airflow_cfm=1800.0,
            coil_capacity_tons=4.5,
        )
        self.assertFalse(res_inv_temp["pass"])
        self.assertEqual(res_inv_temp["reason"], "supply_temp_not_below_room_temp")

        # Negative latent load
        res_neg_latent = verify_hvac_cooling_load_capacity(
            sensible_load_btuh=36000.0,
            latent_load_btuh=-500.0,
            room_temp_db_f=75.0,
            supply_temp_db_f=55.0,
            design_airflow_cfm=1800.0,
            coil_capacity_tons=4.5,
        )
        self.assertFalse(res_neg_latent["pass"])
        self.assertEqual(res_neg_latent["reason"], "negative_latent_load")

        # NaN input
        res_nan = verify_hvac_cooling_load_capacity(
            sensible_load_btuh=float("nan"),
            latent_load_btuh=12000.0,
            room_temp_db_f=75.0,
            supply_temp_db_f=55.0,
            design_airflow_cfm=1800.0,
            coil_capacity_tons=4.5,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")


    def test_aisc_column_base_plate(self) -> None:
        from harness.verifiers import verify_aisc_column_base_plate

        res_pass = verify_aisc_column_base_plate(
            pu_kips=400.0,
            b_in=18.0,
            n_in=18.0,
            tp_in=1.75,
            col_d_in=12.7,
            col_bf_in=12.2,
            fc_psi=4000.0,
            fy_ksi=36.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertAlmostEqual(res_pass["critical_cantilever_in"], 4.12, places=2)
        self.assertAlmostEqual(res_pass["phi_pp_kips"], 716.04, places=1)
        self.assertAlmostEqual(res_pass["t_min_in"], 1.137, places=2)

        res_pedestal = verify_aisc_column_base_plate(
            pu_kips=800.0,
            b_in=18.0,
            n_in=18.0,
            tp_in=2.0,
            col_d_in=12.7,
            col_bf_in=12.2,
            fc_psi=4000.0,
            fy_ksi=36.0,
            pedestal_area_sqin=1296.0,
        )
        self.assertTrue(res_pedestal["pass"])
        self.assertAlmostEqual(res_pedestal["phi_pp_kips"], 1432.08, places=1)

        res_fail_thick = verify_aisc_column_base_plate(
            pu_kips=400.0,
            b_in=18.0,
            n_in=18.0,
            tp_in=1.0,
            col_d_in=12.7,
            col_bf_in=12.2,
            fc_psi=4000.0,
            fy_ksi=36.0,
        )
        self.assertFalse(res_fail_thick["pass"])
        self.assertGreater(res_fail_thick["thickness_dcr"], 1.0)

        res_fail_bearing = verify_aisc_column_base_plate(
            pu_kips=900.0,
            b_in=18.0,
            n_in=18.0,
            tp_in=2.5,
            col_d_in=12.7,
            col_bf_in=12.2,
            fc_psi=4000.0,
            fy_ksi=36.0,
        )
        self.assertFalse(res_fail_bearing["pass"])
        self.assertGreater(res_fail_bearing["bearing_dcr"], 1.0)

        res_col_too_big = verify_aisc_column_base_plate(
            pu_kips=100.0,
            b_in=10.0,
            n_in=18.0,
            tp_in=1.0,
            col_d_in=12.7,
            col_bf_in=12.2,
            fc_psi=4000.0,
        )
        self.assertFalse(res_col_too_big["pass"])
        self.assertEqual(res_col_too_big["reason"], "column_exceeds_base_plate")

        res_nan = verify_aisc_column_base_plate(
            pu_kips=float("nan"),
            b_in=18.0,
            n_in=18.0,
            tp_in=1.75,
            col_d_in=12.7,
            col_bf_in=12.2,
            fc_psi=4000.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aisc_column_base_plate(
            pu_kips=-50.0,
            b_in=18.0,
            n_in=18.0,
            tp_in=1.75,
            col_d_in=12.7,
            col_bf_in=12.2,
            fc_psi=4000.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")


    def test_aci_rebar_development_length(self) -> None:
        from harness.verifiers import verify_aci_rebar_development_length

        # Standard #8 bottom bar passing
        res_pass = verify_aci_rebar_development_length(
            provided_length_in=30.0,
            bar_size_num=8,
            fy_psi=60000.0,
            fc_psi=4000.0,
            confinement_override=2.5,
            casting_position="other",
            coating_type="uncoated",
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertAlmostEqual(res_pass["ld_required_in"], 28.46, places=1)
        self.assertEqual(res_pass["psi_s"], 1.0)
        self.assertEqual(res_pass["psi_g"], 1.0)

        # Top bar factor (psi_t = 1.3) pushes required length over 30 inches
        res_top_fail = verify_aci_rebar_development_length(
            provided_length_in=30.0,
            bar_size_num=8,
            fy_psi=60000.0,
            fc_psi=4000.0,
            confinement_override=2.5,
            casting_position="top",
        )
        self.assertFalse(res_top_fail["pass"])
        self.assertGreater(res_top_fail["dcr"], 1.0)
        self.assertAlmostEqual(res_top_fail["ld_required_in"], 37.0, places=1)

        # 12-inch code minimum development length check for small bar (#4)
        res_min_len = verify_aci_rebar_development_length(
            provided_length_in=14.0,
            bar_size_num=4,
            fy_psi=60000.0,
            fc_psi=4000.0,
            confinement_override=2.5,
        )
        self.assertTrue(res_min_len["pass"])
        self.assertEqual(res_min_len["ld_required_in"], 12.0)
        self.assertLess(res_min_len["ld_calc_in"], 12.0)
        self.assertEqual(res_min_len["psi_s"], 0.8)

        # Epoxy coating cap test: psi_t * psi_e = 1.3 * 1.5 = 1.95 capped at 1.7
        res_epoxy_cap = verify_aci_rebar_development_length(
            provided_length_in=60.0,
            bar_size_num=8,
            fy_psi=60000.0,
            fc_psi=4000.0,
            confinement_override=2.5,
            casting_position="top",
            coating_type="epoxy_low_cover",
        )
        self.assertEqual(res_epoxy_cap["psi_t_e"], 1.7)

        # Error handling tests
        res_nan = verify_aci_rebar_development_length(
            provided_length_in=float("nan"),
            bar_size_num=8,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aci_rebar_development_length(
            provided_length_in=-10.0,
            bar_size_num=8,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_provided_length")

        res_inv_bar = verify_aci_rebar_development_length(
            provided_length_in=30.0,
            bar_size_num=99,
        )
        self.assertFalse(res_inv_bar["pass"])
        self.assertEqual(res_inv_bar["reason"], "invalid_bar_size")


    def test_asce7_roof_snow_load(self) -> None:
        from harness.verifiers import verify_asce7_roof_snow_load

        # Flat roof governed by pf
        res_pass = verify_asce7_roof_snow_load(
            pg_psf=30.0,
            roof_slope_deg=0.0,
            p_capacity_psf=30.0,
            ce_exposure_factor=1.0,
            ct_thermal_factor=1.0,
            importance_factor=1.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertAlmostEqual(res_pass["pf_psf"], 21.0, places=1)
        self.assertAlmostEqual(res_pass["p_design_psf"], 21.0, places=1)
        self.assertAlmostEqual(res_pass["dcr"], 0.70, places=2)
        self.assertEqual(res_pass["cs"], 1.0)
        self.assertTrue(res_pass["is_low_slope"])

        # Low ground snow load where minimum load pm governs
        res_pm_gov = verify_asce7_roof_snow_load(
            pg_psf=15.0,
            roof_slope_deg=0.0,
            p_capacity_psf=20.0,
        )
        self.assertTrue(res_pm_gov["pass"])
        self.assertAlmostEqual(res_pm_gov["pf_psf"], 10.5, places=1)
        self.assertAlmostEqual(res_pm_gov["pm_psf"], 15.0, places=1)
        self.assertAlmostEqual(res_pm_gov["p_design_psf"], 15.0, places=1)
        self.assertAlmostEqual(res_pm_gov["dcr"], 0.75, places=2)

        # Sloped roof with slope reduction factor Cs
        res_sloped = verify_asce7_roof_snow_load(
            pg_psf=30.0,
            roof_slope_deg=50.0,
            p_capacity_psf=15.0,
            is_slippery_surface=False,
        )
        self.assertTrue(res_sloped["pass"])
        self.assertAlmostEqual(res_sloped["cs"], 0.50, places=2)
        self.assertAlmostEqual(res_sloped["ps_psf"], 10.5, places=1)
        self.assertAlmostEqual(res_sloped["p_design_psf"], 10.5, places=1)
        self.assertFalse(res_sloped["is_low_slope"])

        # Steep roof shedding snow completely (Cs = 0)
        res_steep = verify_asce7_roof_snow_load(
            pg_psf=40.0,
            roof_slope_deg=75.0,
            p_capacity_psf=10.0,
        )
        self.assertTrue(res_steep["pass"])
        self.assertEqual(res_steep["cs"], 0.0)
        self.assertEqual(res_steep["p_design_psf"], 0.0)

        # Capacity exceeded failure
        res_fail = verify_asce7_roof_snow_load(
            pg_psf=50.0,
            roof_slope_deg=0.0,
            p_capacity_psf=25.0,
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr"], 1.0)

        # Input error validation
        res_nan = verify_asce7_roof_snow_load(
            pg_psf=float("nan"),
            roof_slope_deg=10.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_asce7_roof_snow_load(
            pg_psf=-20.0,
            roof_slope_deg=10.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_loading_or_capacity")

        res_inv_slope = verify_asce7_roof_snow_load(
            pg_psf=30.0,
            roof_slope_deg=105.0,
        )
        self.assertFalse(res_inv_slope["pass"])
        self.assertEqual(res_inv_slope["reason"], "invalid_roof_slope")


    def test_aisc_beam_column_interaction(self) -> None:
        from harness.verifiers import verify_aisc_beam_column_interaction

        # Passing case governed by Eq. H1-1a (P_u / phi_Pn >= 0.2)
        res_h1a = verify_aisc_beam_column_interaction(
            pu_kips=150.0,
            mux_kip_ft=80.0,
            muy_kip_ft=15.0,
            phi_pn_kips=500.0,
            phi_mnx_kip_ft=250.0,
            phi_mny_kip_ft=60.0,
        )
        self.assertTrue(res_h1a["pass"])
        self.assertEqual(res_h1a["governing_equation"], "H1-1a")
        self.assertAlmostEqual(res_h1a["axial_ratio"], 0.30, places=2)
        self.assertAlmostEqual(res_h1a["dcr"], 0.8067, places=3)

        # Passing case governed by Eq. H1-1b (P_u / phi_Pn < 0.2)
        res_h1b = verify_aisc_beam_column_interaction(
            pu_kips=50.0,
            mux_kip_ft=180.0,
            muy_kip_ft=10.0,
            phi_pn_kips=500.0,
            phi_mnx_kip_ft=250.0,
            phi_mny_kip_ft=60.0,
        )
        self.assertTrue(res_h1b["pass"])
        self.assertEqual(res_h1b["governing_equation"], "H1-1b")
        self.assertAlmostEqual(res_h1b["axial_ratio"], 0.10, places=2)
        self.assertAlmostEqual(res_h1b["dcr"], 0.9367, places=3)

        # B1 moment magnification check
        res_b1 = verify_aisc_beam_column_interaction(
            pu_kips=200.0,
            mux_kip_ft=80.0,
            muy_kip_ft=10.0,
            phi_pn_kips=600.0,
            phi_mnx_kip_ft=300.0,
            phi_mny_kip_ft=80.0,
            cm_x=1.0,
            pe1_x_kips=1000.0,
        )
        self.assertTrue(res_b1["pass"])
        self.assertAlmostEqual(res_b1["b1_x"], 1.25, places=2)
        self.assertAlmostEqual(res_b1["m_rx_kip_ft"], 100.0, places=1)

        # Overloaded interaction check failure
        res_fail = verify_aisc_beam_column_interaction(
            pu_kips=400.0,
            mux_kip_ft=200.0,
            muy_kip_ft=50.0,
            phi_pn_kips=500.0,
            phi_mnx_kip_ft=250.0,
            phi_mny_kip_ft=60.0,
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr"], 1.0)

        # Euler buckling instability error
        res_euler = verify_aisc_beam_column_interaction(
            pu_kips=600.0,
            mux_kip_ft=50.0,
            muy_kip_ft=10.0,
            phi_pn_kips=800.0,
            phi_mnx_kip_ft=300.0,
            phi_mny_kip_ft=100.0,
            pe1_x_kips=500.0,
        )
        self.assertFalse(res_euler["pass"])
        self.assertEqual(res_euler["reason"], "axial_load_exceeds_euler_buckling_x")

        # Input error handling
        res_nan = verify_aisc_beam_column_interaction(
            pu_kips=float("nan"),
            mux_kip_ft=50.0,
            muy_kip_ft=10.0,
            phi_pn_kips=500.0,
            phi_mnx_kip_ft=200.0,
            phi_mny_kip_ft=50.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aisc_beam_column_interaction(
            pu_kips=-10.0,
            mux_kip_ft=50.0,
            muy_kip_ft=10.0,
            phi_pn_kips=500.0,
            phi_mnx_kip_ft=200.0,
            phi_mny_kip_ft=50.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")


    def test_aci_column_axial_load(self) -> None:
        from harness.verifiers import verify_aci_column_axial_load

        # Standard tied column passing
        res_tied = verify_aci_column_axial_load(
            pu_kips=650.0,
            ag_in2=324.0,
            ast_in2=8.0,
            fc_psi=4000.0,
            fy_psi=60000.0,
            transverse_type="tied",
        )
        self.assertTrue(res_tied["pass"])
        self.assertAlmostEqual(res_tied["rho_g"], 8.0 / 324.0, places=4)
        self.assertAlmostEqual(res_tied["p0_kips"], 1554.4, places=1)
        self.assertAlmostEqual(res_tied["phi_pn_max_kips"], 808.29, places=1)
        self.assertLess(res_tied["dcr"], 1.0)
        self.assertEqual(res_tied["phi"], 0.65)

        # Spiral column passing with confinement verification
        res_spiral = verify_aci_column_axial_load(
            pu_kips=700.0,
            ag_in2=314.16,
            ast_in2=6.0,
            fc_psi=4000.0,
            fy_psi=60000.0,
            transverse_type="spiral",
            core_diameter_in=17.0,
            spiral_pitch_in=2.0,
            spiral_bar_dia_in=0.375,
            fyt_psi=60000.0,
        )
        self.assertTrue(res_spiral["pass"])
        self.assertEqual(res_spiral["phi"], 0.75)
        self.assertAlmostEqual(res_spiral["p0_kips"], 1407.74, places=1)
        self.assertAlmostEqual(res_spiral["phi_pn_max_kips"], 897.44, places=1)
        self.assertLess(res_spiral["dcr"], 1.0)
        self.assertGreaterEqual(res_spiral["rho_s_provided"], res_spiral["rho_s_min"])

        # Overloaded tied column failure
        res_fail_load = verify_aci_column_axial_load(
            pu_kips=900.0,
            ag_in2=324.0,
            ast_in2=8.0,
            fc_psi=4000.0,
            transverse_type="tied",
        )
        self.assertFalse(res_fail_load["pass"])
        self.assertGreater(res_fail_load["dcr"], 1.0)

        # Longitudinal steel ratio below minimum (rho_g < 0.01)
        res_low_steel = verify_aci_column_axial_load(
            pu_kips=200.0,
            ag_in2=400.0,
            ast_in2=2.0,
            fc_psi=4000.0,
        )
        self.assertFalse(res_low_steel["pass"])
        self.assertEqual(res_low_steel["reason"], "reinforcement_ratio_out_of_bounds")

        # Spiral pitch spacing out of bounds (> 3.0 in clear spacing)
        res_bad_pitch = verify_aci_column_axial_load(
            pu_kips=500.0,
            ag_in2=314.16,
            ast_in2=6.0,
            fc_psi=4000.0,
            transverse_type="spiral",
            core_diameter_in=17.0,
            spiral_pitch_in=4.5,
            spiral_bar_dia_in=0.375,
        )
        self.assertFalse(res_bad_pitch["pass"])

        # Input error checks
        res_nan = verify_aci_column_axial_load(
            pu_kips=float("nan"),
            ag_in2=324.0,
            ast_in2=8.0,
            fc_psi=4000.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aci_column_axial_load(
            pu_kips=-100.0,
            ag_in2=324.0,
            ast_in2=8.0,
            fc_psi=4000.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")

        res_inv_type = verify_aci_column_axial_load(
            pu_kips=500.0,
            ag_in2=324.0,
            ast_in2=8.0,
            fc_psi=4000.0,
            transverse_type="bundled",
        )
        self.assertFalse(res_inv_type["pass"])
        self.assertEqual(res_inv_type["reason"], "invalid_transverse_type")


    def test_darcy_weisbach_pipe_flow(self) -> None:
        from harness.verifiers import verify_darcy_weisbach_pipe_flow

        # Standard turbulent water pipe passing case
        res_pass = verify_darcy_weisbach_pipe_flow(
            flow_rate_gpm=150.0,
            diameter_in=4.026,
            pipe_length_ft=200.0,
            roughness_in=0.0018,
            max_velocity_fps=8.0,
            max_pressure_drop_psi=5.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertEqual(res_pass["flow_regime"], "turbulent")
        self.assertGreater(res_pass["reynolds_number"], 4000.0)
        self.assertAlmostEqual(res_pass["velocity_fps"], 3.78, places=1)
        self.assertLess(res_pass["pressure_drop_psi"], 2.0)

        # Laminar flow branch check with viscous fluid
        res_laminar = verify_darcy_weisbach_pipe_flow(
            flow_rate_gpm=5.0,
            diameter_in=2.0,
            pipe_length_ft=50.0,
            kinematic_viscosity_ft2_s=1.0e-3,
            max_velocity_fps=5.0,
            max_pressure_drop_psi=10.0,
        )
        self.assertTrue(res_laminar["pass"])
        self.assertEqual(res_laminar["flow_regime"], "laminar")
        self.assertLessEqual(res_laminar["reynolds_number"], 2000.0)
        self.assertAlmostEqual(
            res_laminar["friction_factor"],
            64.0 / res_laminar["reynolds_number"],
            places=4,
        )

        # Minor loss addition check
        res_minor = verify_darcy_weisbach_pipe_flow(
            flow_rate_gpm=150.0,
            diameter_in=4.026,
            pipe_length_ft=200.0,
            minor_loss_k=10.0,
        )
        self.assertTrue(res_minor["pass"])
        self.assertGreater(res_minor["head_loss_minor_ft"], 0.0)
        self.assertAlmostEqual(
            res_minor["total_head_loss_ft"],
            res_minor["head_loss_friction_ft"] + res_minor["head_loss_minor_ft"],
            places=3,
        )

        # Velocity limit exceeded failure
        res_vel_fail = verify_darcy_weisbach_pipe_flow(
            flow_rate_gpm=500.0,
            diameter_in=4.026,
            pipe_length_ft=100.0,
            max_velocity_fps=10.0,
        )
        self.assertFalse(res_vel_fail["pass"])
        self.assertGreater(res_vel_fail["dcr"], 1.0)

        # Input validation checks
        res_nan = verify_darcy_weisbach_pipe_flow(
            flow_rate_gpm=float("nan"),
            diameter_in=4.0,
            pipe_length_ft=100.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_darcy_weisbach_pipe_flow(
            flow_rate_gpm=-50.0,
            diameter_in=4.0,
            pipe_length_ft=100.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_flow_rate")

        res_inv_k = verify_darcy_weisbach_pipe_flow(
            flow_rate_gpm=50.0,
            diameter_in=4.0,
            pipe_length_ft=100.0,
            minor_loss_k=-2.0,
        )
        self.assertFalse(res_inv_k["pass"])
        self.assertEqual(res_inv_k["reason"], "negative_minor_loss")


    def test_asce7_story_drift_and_pdelta(self) -> None:
        from harness.verifiers import verify_asce7_story_drift_and_pdelta

        # Standard passing case without P-delta magnification (theta <= 0.10)
        res_pass = verify_asce7_story_drift_and_pdelta(
            story_height_in=144.0,
            elastic_drift_in=0.35,
            cd_factor=4.0,
            importance_factor_ie=1.0,
            risk_category="II",
            total_vertical_load_px_kips=2000.0,
            story_shear_vx_kips=120.0,
            shear_demand_capacity_ratio_beta=1.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertAlmostEqual(res_pass["design_drift_in"], 1.40, places=2)
        self.assertAlmostEqual(res_pass["allowable_drift_in"], 2.88, places=2)
        self.assertAlmostEqual(res_pass["stability_coeff_theta"], 0.0405, places=4)
        self.assertEqual(res_pass["theta_max"], 0.125)
        self.assertFalse(res_pass["pdelta_required"])
        self.assertEqual(res_pass["pdelta_amplifier"], 1.0)
        self.assertLess(res_pass["dcr"], 1.0)

        # P-delta amplification required (0.10 < theta <= theta_max)
        res_pdelta_amp = verify_asce7_story_drift_and_pdelta(
            story_height_in=144.0,
            elastic_drift_in=0.40,
            cd_factor=4.0,
            importance_factor_ie=1.0,
            risk_category="II",
            total_vertical_load_px_kips=3600.0,
            story_shear_vx_kips=90.0,
        )
        self.assertTrue(res_pdelta_amp["pass"])
        self.assertTrue(res_pdelta_amp["pdelta_required"])
        self.assertAlmostEqual(res_pdelta_amp["stability_coeff_theta"], 0.1111, places=3)
        self.assertGreater(res_pdelta_amp["pdelta_amplifier"], 1.0)
        self.assertAlmostEqual(res_pdelta_amp["pdelta_amplifier"], 1.125, places=2)
        self.assertAlmostEqual(res_pdelta_amp["design_drift_in"], 1.80, places=2)

        # Story drift limit exceeded failure (dcr > 1.0)
        res_fail_drift = verify_asce7_story_drift_and_pdelta(
            story_height_in=144.0,
            elastic_drift_in=0.85,
            cd_factor=4.0,
            risk_category="II",
        )
        self.assertFalse(res_fail_drift["pass"])
        self.assertGreater(res_fail_drift["dcr"], 1.0)

        # Stability coefficient exceeds theta_max (structural instability)
        res_fail_stability = verify_asce7_story_drift_and_pdelta(
            story_height_in=144.0,
            elastic_drift_in=0.30,
            cd_factor=4.0,
            risk_category="II",
            total_vertical_load_px_kips=6000.0,
            story_shear_vx_kips=80.0,
        )
        self.assertFalse(res_fail_stability["pass"])
        self.assertFalse(res_fail_stability["stability_ok"])
        self.assertGreater(res_fail_stability["stability_coeff_theta"], res_fail_stability["theta_max"])

        # Input error validation
        res_nan = verify_asce7_story_drift_and_pdelta(
            story_height_in=float("nan"),
            elastic_drift_in=0.30,
            cd_factor=4.0,
        )
        self.assertFalse(res_nan["pass"]); self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_asce7_story_drift_and_pdelta(
            story_height_in=144.0,
            elastic_drift_in=-0.10,
            cd_factor=4.0,
        )
        self.assertFalse(res_neg["pass"]); self.assertEqual(res_neg["reason"], "negative_drift_input")

        res_inv_cat = verify_asce7_story_drift_and_pdelta(
            story_height_in=144.0,
            elastic_drift_in=0.30,
            cd_factor=4.0,
            risk_category="V",
        )
        self.assertFalse(res_inv_cat["pass"]); self.assertEqual(res_inv_cat["reason"], "invalid_risk_category")


    def test_nec_grounding_electrode_conductor(self) -> None:
        from harness.verifiers import verify_nec_grounding_electrode_conductor

        # 500 kcmil Cu service conductor with 1/0 AWG Cu GEC (105.6 kcmil) -> Pass
        res_pass = verify_nec_grounding_electrode_conductor(
            service_conductor_kcmil=500.0,
            gec_provided_kcmil=105.6,
            service_material="copper",
            gec_material="copper",
        )
        self.assertTrue(res_pass["pass"])
        self.assertAlmostEqual(res_pass["dcr"], 1.0, places=3)
        self.assertEqual(res_pass["min_gec_kcmil"], 105.6)

        # 500 kcmil Cu service conductor with undersized 2 AWG Cu GEC (66.36 kcmil) -> Fail
        res_fail = verify_nec_grounding_electrode_conductor(
            service_conductor_kcmil=500.0,
            gec_provided_kcmil=66.36,
            service_material="copper",
            gec_material="copper",
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr"], 1.0)

        # Aluminum service conductor (750 kcmil Al) with Cu GEC (1/0 Cu = 105.6 kcmil) -> Pass
        res_al_service = verify_nec_grounding_electrode_conductor(
            service_conductor_kcmil=750.0,
            gec_provided_kcmil=105.6,
            service_material="aluminum",
            gec_material="copper",
        )
        self.assertTrue(res_al_service["pass"])
        self.assertEqual(res_al_service["min_gec_kcmil"], 105.6)

        # Large Cu service (>1100 kcmil: 2000 kcmil) 12.5% rule
        res_large = verify_nec_grounding_electrode_conductor(
            service_conductor_kcmil=2000.0,
            gec_provided_kcmil=250.0,
            service_material="copper",
            gec_material="copper",
        )
        self.assertTrue(res_large["pass"])
        self.assertEqual(res_large["min_gec_kcmil"], 250.0)

        # Invalid material check
        res_mat = verify_nec_grounding_electrode_conductor(
            service_conductor_kcmil=500.0,
            gec_provided_kcmil=105.6,
            service_material="silver",
        )
        self.assertFalse(res_mat["pass"])
        self.assertEqual(res_mat["reason"], "invalid_conductor_material")

        # NaN and negative edge cases
        res_nan = verify_nec_grounding_electrode_conductor(
            service_conductor_kcmil=float("nan"),
            gec_provided_kcmil=105.6,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_nec_grounding_electrode_conductor(
            service_conductor_kcmil=-500.0,
            gec_provided_kcmil=105.6,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_conductor_area")


    def test_aashto_stopping_sight_distance(self) -> None:
        from harness.verifiers import verify_aashto_stopping_sight_distance

        # Standard level roadway passing SSD and crest vertical curve check
        res_pass = verify_aashto_stopping_sight_distance(
            design_speed_mph=60.0,
            provided_sight_distance_ft=600.0,
            grade_pct=0.0,
            g1_pct=3.0,
            g2_pct=-2.0,
            curve_length_ft=800.0,
            curve_type="crest",
        )
        self.assertTrue(res_pass["pass"])
        self.assertAlmostEqual(res_pass["d_reaction_ft"], 220.05, places=1)
        self.assertAlmostEqual(res_pass["d_braking_ft"], 344.73, places=1)
        self.assertAlmostEqual(res_pass["ssd_required_ft"], 564.78, places=1)
        self.assertAlmostEqual(res_pass["ssd_dcr"], 564.78 / 600.0, places=3)
        self.assertLessEqual(res_pass["l_min_ft"], 800.0)
        self.assertAlmostEqual(res_pass["k_value"], 160.0, places=1)

        # Downgrade condition increasing braking distance and failing SSD
        res_fail_ssd = verify_aashto_stopping_sight_distance(
            design_speed_mph=60.0,
            provided_sight_distance_ft=550.0,
            grade_pct=-4.0,
        )
        self.assertFalse(res_fail_ssd["pass"])
        self.assertGreater(res_fail_ssd["ssd_dcr"], 1.0)
        self.assertGreater(res_fail_ssd["ssd_required_ft"], 600.0)

        # Sag vertical curve passing check
        res_sag = verify_aashto_stopping_sight_distance(
            design_speed_mph=50.0,
            provided_sight_distance_ft=450.0,
            grade_pct=0.0,
            g1_pct=-3.0,
            g2_pct=2.0,
            curve_length_ft=500.0,
            curve_type="sag",
        )
        self.assertTrue(res_sag["pass"])
        self.assertLess(res_sag["curve_dcr"], 1.0)

        # Excessive steep downgrade causing physical inability to stop
        res_steep = verify_aashto_stopping_sight_distance(
            design_speed_mph=45.0,
            provided_sight_distance_ft=500.0,
            grade_pct=-36.0,
        )
        self.assertFalse(res_steep["pass"])
        self.assertEqual(res_steep["reason"], "excessive_downgrade_cannot_stop")

        # Input error validation
        res_nan = verify_aashto_stopping_sight_distance(
            design_speed_mph=float("nan"),
            provided_sight_distance_ft=500.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aashto_stopping_sight_distance(
            design_speed_mph=-50.0,
            provided_sight_distance_ft=500.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_speed_or_geometry")

        res_inv_type = verify_aashto_stopping_sight_distance(
            design_speed_mph=50.0,
            provided_sight_distance_ft=500.0,
            curve_type="spiral",
        )
        self.assertFalse(res_inv_type["pass"])
        self.assertEqual(res_inv_type["reason"], "invalid_curve_type")


    def test_aisc_unbraced_beam_flexure(self) -> None:
        from harness.verifiers import verify_aisc_unbraced_beam_flexure

        # Passing case with short unbraced length governed by plastic yielding (Lb <= Lp)
        res_yield = verify_aisc_unbraced_beam_flexure(
            mu_kip_ft=350.0,
            unbraced_length_ft=8.0,
            zx_in3=115.0,
            sx_in3=103.0,
            ry_in=2.46,
            rts_in=2.80,
            ho_in=13.3,
            j_in4=3.01,
            fy_ksi=50.0,
        )
        self.assertTrue(res_yield["pass"])
        self.assertEqual(res_yield["governing_limit_state"], "plastic_yielding")
        self.assertAlmostEqual(res_yield["phi_mn_kip_ft"], 431.25, places=1)
        self.assertLess(res_yield["dcr"], 1.0)

        # Inelastic LTB check with intermediate unbraced length (Lp < Lb <= Lr)
        res_inelastic = verify_aisc_unbraced_beam_flexure(
            mu_kip_ft=320.0,
            unbraced_length_ft=18.0,
            zx_in3=115.0,
            sx_in3=103.0,
            ry_in=2.46,
            rts_in=2.80,
            ho_in=13.3,
            j_in4=3.01,
            cb=1.14,
        )
        self.assertTrue(res_inelastic["pass"])
        self.assertEqual(res_inelastic["governing_limit_state"], "inelastic_ltb")
        self.assertLess(res_inelastic["phi_mn_kip_ft"], res_yield["phi_mn_kip_ft"])

        # Elastic LTB check with long unbraced length (Lb > Lr) causing failure
        res_elastic_fail = verify_aisc_unbraced_beam_flexure(
            mu_kip_ft=300.0,
            unbraced_length_ft=35.0,
            zx_in3=115.0,
            sx_in3=103.0,
            ry_in=2.46,
            rts_in=2.80,
            ho_in=13.3,
            j_in4=3.01,
            cb=1.0,
        )
        self.assertFalse(res_elastic_fail["pass"])
        self.assertEqual(res_elastic_fail["governing_limit_state"], "elastic_ltb")
        self.assertGreater(res_elastic_fail["dcr"], 1.0)

        # Edge case: Section modulus inconsistency (Sx > Zx)
        res_sx_err = verify_aisc_unbraced_beam_flexure(
            mu_kip_ft=100.0,
            unbraced_length_ft=10.0,
            zx_in3=80.0,
            sx_in3=100.0,
            ry_in=2.0,
            rts_in=2.5,
            ho_in=12.0,
            j_in4=2.0,
        )
        self.assertFalse(res_sx_err["pass"])
        self.assertEqual(res_sx_err["reason"], "sx_exceeds_zx")

        # Edge case: Negative moment load
        res_neg = verify_aisc_unbraced_beam_flexure(
            mu_kip_ft=-50.0,
            unbraced_length_ft=10.0,
            zx_in3=115.0,
            sx_in3=103.0,
            ry_in=2.46,
            rts_in=2.80,
            ho_in=13.3,
            j_in4=3.01,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")

        # Edge case: NaN input handling
        res_nan = verify_aisc_unbraced_beam_flexure(
            mu_kip_ft=float("nan"),
            unbraced_length_ft=10.0,
            zx_in3=115.0,
            sx_in3=103.0,
            ry_in=2.46,
            rts_in=2.80,
            ho_in=13.3,
            j_in4=3.01,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")


    def test_aisc_web_local_yielding_and_crippling(self) -> None:
        from harness.verifiers import verify_aisc_web_local_yielding_and_crippling

        # End bearing support reaction passing without stiffeners (W18x50 parameters)
        res_end_pass = verify_aisc_web_local_yielding_and_crippling(
            pu_kips=60.0,
            d_in=18.0,
            tw_in=0.355,
            tf_in=0.570,
            k_des_in=0.98,
            lb_bearing_in=4.0,
            distance_to_end_in=0.0,
            fy_ksi=50.0,
        )
        self.assertTrue(res_end_pass["pass"])
        self.assertFalse(res_end_pass["stiffeners_required"])
        self.assertEqual(res_end_pass["governing_limit_state"], "web_local_crippling")
        self.assertAlmostEqual(res_end_pass["phi_rn_yielding_kips"], 114.49, places=1)
        self.assertAlmostEqual(res_end_pass["phi_rn_crippling_kips"], 77.23, places=1)
        self.assertAlmostEqual(res_end_pass["dcr"], 60.0 / 77.23, places=3)

        # Interior concentrated load passing (distance > d)
        res_interior_pass = verify_aisc_web_local_yielding_and_crippling(
            pu_kips=120.0,
            d_in=18.0,
            tw_in=0.355,
            tf_in=0.570,
            k_des_in=0.98,
            lb_bearing_in=4.0,
            distance_to_end_in=60.0,
            fy_ksi=50.0,
        )
        self.assertTrue(res_interior_pass["pass"])
        self.assertEqual(res_interior_pass["governing_limit_state"], "web_local_crippling")
        self.assertAlmostEqual(res_interior_pass["phi_rn_yielding_kips"], 157.98, places=1)
        self.assertAlmostEqual(res_interior_pass["phi_rn_crippling_kips"], 153.19, places=1)

        # End reaction overload requiring stiffeners (dcr > 1.0)
        res_end_fail = verify_aisc_web_local_yielding_and_crippling(
            pu_kips=95.0,
            d_in=18.0,
            tw_in=0.355,
            tf_in=0.570,
            k_des_in=0.98,
            lb_bearing_in=4.0,
            distance_to_end_in=0.0,
        )
        self.assertFalse(res_end_fail["pass"])
        self.assertTrue(res_end_fail["stiffeners_required"])
        self.assertGreater(res_end_fail["dcr"], 1.0)

        # Short bearing length at end (lb / d <= 0.20)
        res_short_bearing = verify_aisc_web_local_yielding_and_crippling(
            pu_kips=50.0,
            d_in=18.0,
            tw_in=0.355,
            tf_in=0.570,
            k_des_in=0.98,
            lb_bearing_in=3.0,
            distance_to_end_in=0.0,
        )
        self.assertTrue(res_short_bearing["pass"])
        self.assertAlmostEqual(res_short_bearing["phi_rn_crippling_kips"], 71.87, places=1)

        # Input error validation checks
        res_nan = verify_aisc_web_local_yielding_and_crippling(
            pu_kips=float("nan"),
            d_in=18.0,
            tw_in=0.355,
            tf_in=0.570,
            k_des_in=0.98,
            lb_bearing_in=4.0,
            distance_to_end_in=0.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg_load = verify_aisc_web_local_yielding_and_crippling(
            pu_kips=-50.0,
            d_in=18.0,
            tw_in=0.355,
            tf_in=0.570,
            k_des_in=0.98,
            lb_bearing_in=4.0,
            distance_to_end_in=0.0,
        )
        self.assertFalse(res_neg_load["pass"])
        self.assertEqual(res_neg_load["reason"], "negative_load_input")

        res_k_err = verify_aisc_web_local_yielding_and_crippling(
            pu_kips=50.0,
            d_in=18.0,
            tw_in=0.355,
            tf_in=0.570,
            k_des_in=20.0,
            lb_bearing_in=4.0,
            distance_to_end_in=0.0,
        )
        self.assertFalse(res_k_err["pass"])
        self.assertEqual(res_k_err["reason"], "flange_or_k_exceeds_depth")


    def test_asce7_seismic_force_distribution(self) -> None:
        from harness.verifiers import verify_asce7_seismic_force_distribution

        # Standard 3-story building passing case
        res_pass = verify_asce7_seismic_force_distribution(
            story_weights_kips=[1000.0, 1000.0, 800.0],
            story_elevations_ft=[14.0, 28.0, 42.0],
            story_shear_capacities_kips=[250.0, 200.0, 120.0],
            base_shear_kips=200.0,
            fundamental_period_s=1.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["k_exponent"], 1.25)
        self.assertEqual(len(res_pass["story_forces_kips"]), 3)
        self.assertAlmostEqual(res_pass["story_shears_kips"][0], 200.0, places=1)
        self.assertEqual(res_pass["governing_story"], 2)
        self.assertAlmostEqual(res_pass["dcr"], 0.8471, places=3)

        # Capacity exceeded at upper story failure
        res_fail = verify_asce7_seismic_force_distribution(
            story_weights_kips=[1000.0, 1000.0, 800.0],
            story_elevations_ft=[14.0, 28.0, 42.0],
            story_shear_capacities_kips=[250.0, 200.0, 80.0],
            base_shear_kips=200.0,
            fundamental_period_s=1.0,
        )
        self.assertFalse(res_fail["pass"])
        self.assertEqual(res_fail["governing_story"], 3)
        self.assertGreater(res_fail["dcr"], 1.0)

        # Short period k = 1.0
        res_short = verify_asce7_seismic_force_distribution(
            story_weights_kips=[500.0, 500.0],
            story_elevations_ft=[12.0, 24.0],
            story_shear_capacities_kips=[150.0, 100.0],
            base_shear_kips=100.0,
            fundamental_period_s=0.4,
        )
        self.assertTrue(res_short["pass"])
        self.assertEqual(res_short["k_exponent"], 1.0)

        # Long period k = 2.0
        res_long = verify_asce7_seismic_force_distribution(
            story_weights_kips=[500.0, 500.0],
            story_elevations_ft=[12.0, 24.0],
            story_shear_capacities_kips=[150.0, 100.0],
            base_shear_kips=100.0,
            fundamental_period_s=3.0,
        )
        self.assertTrue(res_long["pass"])
        self.assertEqual(res_long["k_exponent"], 2.0)

        # Edge cases: Mismatched list lengths
        res_mismatch = verify_asce7_seismic_force_distribution(
            story_weights_kips=[1000.0, 1000.0],
            story_elevations_ft=[14.0],
            story_shear_capacities_kips=[200.0, 200.0],
            base_shear_kips=100.0,
            fundamental_period_s=0.5,
        )
        self.assertFalse(res_mismatch["pass"])
        self.assertEqual(res_mismatch["reason"], "mismatched_story_data")

        # Edge case: Non-increasing elevations
        res_elev_err = verify_asce7_seismic_force_distribution(
            story_weights_kips=[1000.0, 1000.0],
            story_elevations_ft=[20.0, 15.0],
            story_shear_capacities_kips=[200.0, 200.0],
            base_shear_kips=100.0,
            fundamental_period_s=0.5,
        )
        self.assertFalse(res_elev_err["pass"])
        self.assertEqual(res_elev_err["reason"], "non_increasing_elevations")

        # Edge case: NaN input
        res_nan = verify_asce7_seismic_force_distribution(
            story_weights_kips=[float("nan"), 1000.0],
            story_elevations_ft=[14.0, 28.0],
            story_shear_capacities_kips=[200.0, 200.0],
            base_shear_kips=100.0,
            fundamental_period_s=0.5,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")


    def test_asme_b313_pipe_thickness(self) -> None:
        from harness.verifiers import verify_asme_b313_pipe_thickness

        # 6-inch NPS Schedule 40 carbon steel pipe passing at 500 psi
        res_pass = verify_asme_b313_pipe_thickness(
            design_pressure_psi=500.0,
            outer_diameter_in=6.625,
            nominal_wall_thickness_in=0.280,
            allowable_stress_psi=20000.0,
            weld_efficiency_e=1.0,
            corrosion_allowance_in=0.0625,
            mechanical_allowance_in=0.0,
            mill_tolerance_pct=12.5,
        )
        self.assertTrue(res_pass["pass"])
        self.assertAlmostEqual(res_pass["t_pressure_in"], 0.082, places=3)
        self.assertAlmostEqual(res_pass["t_nom_required_in"], 0.1651, places=3)
        self.assertAlmostEqual(res_pass["dcr"], 0.5898, places=3)
        self.assertGreater(res_pass["mawp_psi"], 500.0)

        # Overpressured case (1500 psi on Schedule 40) failing thickness requirement
        res_fail_press = verify_asme_b313_pipe_thickness(
            design_pressure_psi=1500.0,
            outer_diameter_in=6.625,
            nominal_wall_thickness_in=0.280,
            allowable_stress_psi=20000.0,
        )
        self.assertFalse(res_fail_press["pass"])
        self.assertGreater(res_fail_press["dcr"], 1.0)
        self.assertLess(res_fail_press["mawp_psi"], 1500.0)

        # Excessive corrosion allowance consuming available wall thickness
        res_corr_err = verify_asme_b313_pipe_thickness(
            design_pressure_psi=200.0,
            outer_diameter_in=6.625,
            nominal_wall_thickness_in=0.280,
            corrosion_allowance_in=0.30,
        )
        self.assertFalse(res_corr_err["pass"])
        self.assertEqual(res_corr_err["reason"], "allowances_exceed_available_thickness")

        # Input error validation checks
        res_nan = verify_asme_b313_pipe_thickness(
            design_pressure_psi=float("nan"),
            outer_diameter_in=6.625,
            nominal_wall_thickness_in=0.280,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_asme_b313_pipe_thickness(
            design_pressure_psi=-100.0,
            outer_diameter_in=6.625,
            nominal_wall_thickness_in=0.280,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_pressure_or_dimensions")

        res_inv_e = verify_asme_b313_pipe_thickness(
            design_pressure_psi=500.0,
            outer_diameter_in=6.625,
            nominal_wall_thickness_in=0.280,
            weld_efficiency_e=1.2,
        )
        self.assertFalse(res_inv_e["pass"])
        self.assertEqual(res_inv_e["reason"], "invalid_weld_efficiency")


    def test_meyerhof_bearing_capacity(self) -> None:
        from harness.verifiers import verify_meyerhof_bearing_capacity

        # Standard spread footing in sand/silt passing check
        res_pass = verify_meyerhof_bearing_capacity(
            applied_load_kips=180.0,
            footing_width_ft=6.0,
            footing_length_ft=6.0,
            embedment_depth_ft=4.0,
            soil_cohesion_psf=200.0,
            soil_friction_angle_deg=30.0,
            soil_unit_weight_pcf=115.0,
            min_fs=3.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertGreaterEqual(res_pass["actual_fs"], 3.0)
        self.assertAlmostEqual(res_pass["q_applied_psf"], 5000.0, places=1)
        self.assertGreater(res_pass["q_ultimate_psf"], 30000.0)

        # Overloaded footing failure (dcr > 1.0)
        res_fail_load = verify_meyerhof_bearing_capacity(
            applied_load_kips=500.0,
            footing_width_ft=6.0,
            footing_length_ft=6.0,
            embedment_depth_ft=4.0,
            soil_cohesion_psf=200.0,
            soil_friction_angle_deg=30.0,
            soil_unit_weight_pcf=115.0,
        )
        self.assertFalse(res_fail_load["pass"])
        self.assertGreater(res_fail_load["dcr"], 1.0)

        # High water table derating capacity
        res_water = verify_meyerhof_bearing_capacity(
            applied_load_kips=180.0,
            footing_width_ft=6.0,
            footing_length_ft=6.0,
            embedment_depth_ft=4.0,
            soil_cohesion_psf=200.0,
            soil_friction_angle_deg=30.0,
            soil_unit_weight_pcf=115.0,
            water_table_depth_ft=1.0,
        )
        self.assertLess(res_water["q_ultimate_psf"], res_pass["q_ultimate_psf"])

        # Excessive eccentricity failure
        res_ecc = verify_meyerhof_bearing_capacity(
            applied_load_kips=50.0,
            footing_width_ft=6.0,
            footing_length_ft=6.0,
            embedment_depth_ft=4.0,
            soil_cohesion_psf=200.0,
            soil_friction_angle_deg=30.0,
            soil_unit_weight_pcf=115.0,
            load_eccentricity_b_ft=3.5,
        )
        self.assertFalse(res_ecc["pass"])
        self.assertEqual(res_ecc["reason"], "excessive_eccentricity")

        # Input validation checks
        res_nan = verify_meyerhof_bearing_capacity(
            applied_load_kips=float("nan"),
            footing_width_ft=6.0,
            footing_length_ft=6.0,
            embedment_depth_ft=4.0,
            soil_cohesion_psf=200.0,
            soil_friction_angle_deg=30.0,
            soil_unit_weight_pcf=115.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_meyerhof_bearing_capacity(
            applied_load_kips=-10.0,
            footing_width_ft=6.0,
            footing_length_ft=6.0,
            embedment_depth_ft=4.0,
            soil_cohesion_psf=200.0,
            soil_friction_angle_deg=30.0,
            soil_unit_weight_pcf=115.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")


    def test_aashto_horizontal_curve(self) -> None:
        from harness.verifiers import verify_aashto_horizontal_curve

        # Standard passing highway curve with adequate HSO
        res_pass = verify_aashto_horizontal_curve(
            design_speed_mph=50.0,
            radius_ft=850.0,
            superelevation_pct=6.0,
            max_superelevation_pct=8.0,
            ssd_ft=425.0,
            hso_provided_ft=30.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertAlmostEqual(res_pass["f_max"], 0.14, places=2)
        self.assertAlmostEqual(res_pass["r_min_ft"], 757.58, places=1)
        self.assertAlmostEqual(res_pass["f_demand"], 0.1361, places=3)
        self.assertAlmostEqual(res_pass["hso_required_ft"], 26.42, places=1)
        self.assertLess(res_pass["dcr"], 1.0)

        # Inadequate curve radius failing minimum radius check
        res_fail_r = verify_aashto_horizontal_curve(
            design_speed_mph=60.0,
            radius_ft=1000.0,
            superelevation_pct=6.0,
            max_superelevation_pct=6.0,
        )
        self.assertFalse(res_fail_r["pass"])
        self.assertGreater(res_fail_r["dcr"], 1.0)
        self.assertAlmostEqual(res_fail_r["r_min_ft"], 1333.33, places=1)

        # Insufficient horizontal sightline offset (sight obstruction) failure
        res_fail_hso = verify_aashto_horizontal_curve(
            design_speed_mph=50.0,
            radius_ft=850.0,
            superelevation_pct=6.0,
            ssd_ft=425.0,
            hso_provided_ft=20.0,
        )
        self.assertFalse(res_fail_hso["pass"])
        self.assertGreater(res_fail_hso["hso_dcr"], 1.0)

        # Superelevation exceeding allowable maximum
        res_fail_e = verify_aashto_horizontal_curve(
            design_speed_mph=45.0,
            radius_ft=900.0,
            superelevation_pct=9.0,
            max_superelevation_pct=8.0,
        )
        self.assertFalse(res_fail_e["pass"])
        self.assertGreater(res_fail_e["dcr"], 1.0)

        # Input error validation checks
        res_nan = verify_aashto_horizontal_curve(
            design_speed_mph=float("nan"),
            radius_ft=850.0,
            superelevation_pct=6.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aashto_horizontal_curve(
            design_speed_mph=-50.0,
            radius_ft=850.0,
            superelevation_pct=6.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_speed_or_geometry")


    def test_nec_equipment_grounding_conductor(self) -> None:
        from harness.verifiers import verify_nec_equipment_grounding_conductor

        # Standard 100A OCPD with 8 AWG Cu (16.51 kcmil) -> Pass
        res_pass = verify_nec_equipment_grounding_conductor(
            ocpd_rating_a=100.0,
            egc_provided_kcmil=16.51,
            egc_material="copper",
        )
        self.assertTrue(res_pass["pass"])
        self.assertAlmostEqual(res_pass["dcr"], 1.0, places=3)
        self.assertEqual(res_pass["base_egc_kcmil"], 16.51)
        self.assertEqual(res_pass["upsize_ratio"], 1.0)

        # 100A OCPD with undersized 10 AWG Cu (10.38 kcmil) -> Fail
        res_fail = verify_nec_equipment_grounding_conductor(
            ocpd_rating_a=100.0,
            egc_provided_kcmil=10.38,
            egc_material="copper",
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr"], 1.0)

        # NEC 250.122(B) proportional upsize test: 200A feeder upsized from 167.8 to 250.0 kcmil
        # Base Cu EGC is 6 AWG (26.24 kcmil). Required = 26.24 * (250/167.8) = 39.09 kcmil
        res_upsize_fail = verify_nec_equipment_grounding_conductor(
            ocpd_rating_a=200.0,
            egc_provided_kcmil=26.24,
            actual_ungrounded_kcmil=250.0,
            min_ungrounded_kcmil=167.8,
        )
        self.assertFalse(res_upsize_fail["pass"])
        self.assertAlmostEqual(res_upsize_fail["required_egc_kcmil"], 39.09, places=1)
        self.assertGreater(res_upsize_fail["dcr"], 1.0)

        # Supplying 4 AWG Cu (41.74 kcmil) satisfies upsized requirement
        res_upsize_pass = verify_nec_equipment_grounding_conductor(
            ocpd_rating_a=200.0,
            egc_provided_kcmil=41.74,
            actual_ungrounded_kcmil=250.0,
            min_ungrounded_kcmil=167.8,
        )
        self.assertTrue(res_upsize_pass["pass"])
        self.assertLess(res_upsize_pass["dcr"], 1.0)

        # Aluminum EGC check: 400A OCPD -> 1 AWG Al (83.69 kcmil)
        res_al = verify_nec_equipment_grounding_conductor(
            ocpd_rating_a=400.0,
            egc_provided_kcmil=83.69,
            egc_material="aluminum",
        )
        self.assertTrue(res_al["pass"])
        self.assertEqual(res_al["base_egc_kcmil"], 83.69)

        # Input error validation
        res_nan = verify_nec_equipment_grounding_conductor(
            ocpd_rating_a=float("nan"),
            egc_provided_kcmil=16.51,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_nec_equipment_grounding_conductor(
            ocpd_rating_a=-100.0,
            egc_provided_kcmil=16.51,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_rating_or_area")

        res_mat = verify_nec_equipment_grounding_conductor(
            ocpd_rating_a=100.0,
            egc_provided_kcmil=16.51,
            egc_material="steel",
        )
        self.assertFalse(res_mat["pass"])
        self.assertEqual(res_mat["reason"], "invalid_conductor_material")

        res_table_exceed = verify_nec_equipment_grounding_conductor(
            ocpd_rating_a=8000.0,
            egc_provided_kcmil=1000.0,
        )
        self.assertFalse(res_table_exceed["pass"])
        self.assertEqual(res_table_exceed["reason"], "ocpd_exceeds_code_table")


    def test_asce7_nonstructural_component_seismic(self) -> None:
        from harness.verifiers import verify_asce7_nonstructural_component_seismic

        # Rooftop chiller passing check (unclamped governing)
        res_pass = verify_asce7_nonstructural_component_seismic(
            w_p_kips=10.0,
            s_ds=1.0,
            ap_factor=2.5,
            rp_factor=6.0,
            z_height_ft=40.0,
            h_roof_ft=40.0,
            f_p_capacity_kips=6.0,
            importance_factor_ip=1.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["governing_limit"], "unclamped")
        self.assertAlmostEqual(res_pass["f_p_kips"], 5.0, places=2)
        self.assertAlmostEqual(res_pass["dcr"], 5.0 / 6.0, places=4)
        self.assertAlmostEqual(res_pass["f_pv_kips"], 2.0, places=2)

        # Ground-level equipment governed by lower bound minimum limit
        res_min = verify_asce7_nonstructural_component_seismic(
            w_p_kips=10.0,
            s_ds=1.0,
            ap_factor=1.0,
            rp_factor=2.5,
            z_height_ft=0.0,
            h_roof_ft=50.0,
            f_p_capacity_kips=4.0,
            importance_factor_ip=1.0,
        )
        self.assertTrue(res_min["pass"])
        self.assertEqual(res_min["governing_limit"], "min_limit")
        self.assertAlmostEqual(res_min["f_p_kips"], 3.0, places=2)
        self.assertAlmostEqual(res_min["f_p_unclamped_kips"], 1.6, places=2)

        # Rooftop flexible component governed by upper bound maximum limit
        res_max = verify_asce7_nonstructural_component_seismic(
            w_p_kips=10.0,
            s_ds=1.2,
            ap_factor=2.5,
            rp_factor=1.5,
            z_height_ft=50.0,
            h_roof_ft=50.0,
            f_p_capacity_kips=35.0,
            importance_factor_ip=1.5,
        )
        self.assertTrue(res_max["pass"])
        self.assertEqual(res_max["governing_limit"], "max_limit")
        self.assertAlmostEqual(res_max["f_p_kips"], 28.8, places=2)
        self.assertGreater(res_max["f_p_unclamped_kips"], 28.8)

        # Capacity exceeded failure
        res_fail = verify_asce7_nonstructural_component_seismic(
            w_p_kips=10.0,
            s_ds=1.0,
            ap_factor=2.5,
            rp_factor=6.0,
            z_height_ft=40.0,
            h_roof_ft=40.0,
            f_p_capacity_kips=4.0,
            importance_factor_ip=1.0,
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr"], 1.0)

        # Height error check (z > h)
        res_height_err = verify_asce7_nonstructural_component_seismic(
            w_p_kips=10.0,
            s_ds=1.0,
            ap_factor=2.5,
            rp_factor=6.0,
            z_height_ft=55.0,
            h_roof_ft=50.0,
            f_p_capacity_kips=6.0,
        )
        self.assertFalse(res_height_err["pass"])
        self.assertEqual(res_height_err["reason"], "attachment_height_exceeds_roof_height")

        # Invalid importance factor
        res_ip_err = verify_asce7_nonstructural_component_seismic(
            w_p_kips=10.0,
            s_ds=1.0,
            ap_factor=2.5,
            rp_factor=6.0,
            z_height_ft=20.0,
            h_roof_ft=40.0,
            f_p_capacity_kips=6.0,
            importance_factor_ip=1.25,
        )
        self.assertFalse(res_ip_err["pass"])
        self.assertEqual(res_ip_err["reason"], "invalid_importance_factor")

        # NaN and negative input validation
        res_nan = verify_asce7_nonstructural_component_seismic(
            w_p_kips=float("nan"),
            s_ds=1.0,
            ap_factor=2.5,
            rp_factor=6.0,
            z_height_ft=20.0,
            h_roof_ft=40.0,
            f_p_capacity_kips=6.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_asce7_nonstructural_component_seismic(
            w_p_kips=-10.0,
            s_ds=1.0,
            ap_factor=2.5,
            rp_factor=6.0,
            z_height_ft=20.0,
            h_roof_ft=40.0,
            f_p_capacity_kips=6.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_loading_or_capacity")


    def test_nfpa13_fire_sprinkler_demand(self) -> None:
        from harness.verifiers import verify_nfpa13_fire_sprinkler_demand

        # Standard wet-pipe Ordinary Hazard Group 1 passing check
        res_pass = verify_nfpa13_fire_sprinkler_demand(
            hazard_classification="ordinary_hazard_group_1",
            coverage_area_per_sprinkler_sqft=130.0,
            available_flow_gpm=500.0,
            available_pressure_psi=75.0,
            system_piping_loss_psi=25.0,
            elevation_loss_psi=10.0,
            k_factor=5.6,
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["num_sprinklers"], 12)
        self.assertEqual(res_pass["design_density_gpm_sqft"], 0.15)
        self.assertAlmostEqual(res_pass["q_sprinklers_gpm"], 234.0, places=1)
        self.assertAlmostEqual(res_pass["q_total_demand_gpm"], 484.0, places=1)
        self.assertAlmostEqual(res_pass["p_head_psi"], 12.13, places=1)
        self.assertAlmostEqual(res_pass["p_total_demand_psi"], 47.13, places=1)
        self.assertAlmostEqual(res_pass["flow_dcr"], 484.0 / 500.0, places=3)
        self.assertLess(res_pass["dcr"], 1.0)

        # Dry pipe system triggers 30% area increase and exceeds flow supply
        res_dry_fail = verify_nfpa13_fire_sprinkler_demand(
            hazard_classification="ordinary_hazard_group_1",
            coverage_area_per_sprinkler_sqft=130.0,
            available_flow_gpm=500.0,
            available_pressure_psi=75.0,
            system_piping_loss_psi=25.0,
            elevation_loss_psi=10.0,
            is_dry_system=True,
        )
        self.assertFalse(res_dry_fail["pass"])
        self.assertEqual(res_dry_fail["design_area_sqft"], 1950.0)
        self.assertEqual(res_dry_fail["num_sprinklers"], 15)
        self.assertGreater(res_dry_fail["q_total_demand_gpm"], 500.0)
        self.assertGreater(res_dry_fail["flow_dcr"], 1.0)

        # Pressure capacity failure (high friction loss)
        res_press_fail = verify_nfpa13_fire_sprinkler_demand(
            hazard_classification="light_hazard",
            coverage_area_per_sprinkler_sqft=150.0,
            available_flow_gpm=500.0,
            available_pressure_psi=40.0,
            system_piping_loss_psi=35.0,
            elevation_loss_psi=15.0,
        )
        self.assertFalse(res_press_fail["pass"])
        self.assertGreater(res_press_fail["pressure_dcr"], 1.0)

        # Sprinkler spacing exceeds maximum allowable per code
        res_spacing_err = verify_nfpa13_fire_sprinkler_demand(
            hazard_classification="extra_hazard_group_1",
            coverage_area_per_sprinkler_sqft=120.0,
            available_flow_gpm=1000.0,
            available_pressure_psi=90.0,
        )
        self.assertFalse(res_spacing_err["pass"])
        self.assertEqual(res_spacing_err["reason"], "sprinkler_coverage_exceeds_maximum")

        # Input error validation
        res_nan = verify_nfpa13_fire_sprinkler_demand(
            hazard_classification="light",
            coverage_area_per_sprinkler_sqft=float("nan"),
            available_flow_gpm=500.0,
            available_pressure_psi=75.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_nfpa13_fire_sprinkler_demand(
            hazard_classification="light",
            coverage_area_per_sprinkler_sqft=150.0,
            available_flow_gpm=-500.0,
            available_pressure_psi=75.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_geometry_or_capacity")

        res_inv_hazard = verify_nfpa13_fire_sprinkler_demand(
            hazard_classification="nuclear_storage",
            coverage_area_per_sprinkler_sqft=100.0,
            available_flow_gpm=500.0,
            available_pressure_psi=75.0,
        )
        self.assertFalse(res_inv_hazard["pass"])
        self.assertEqual(res_inv_hazard["reason"], "invalid_hazard_classification")


    def test_aisc_block_shear(self) -> None:
        from harness.verifiers import verify_aisc_block_shear

        # Standard passing block shear case governed by shear yielding cap
        res_pass = verify_aisc_block_shear(
            pu_kips=80.0,
            agv_in2=4.5,
            anv_in2=3.5,
            ant_in2=1.0,
            fy_ksi=50.0,
            fu_ksi=65.0,
            u_bs=1.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["governing_limit_state"], "shear_yielding")
        self.assertAlmostEqual(res_pass["tension_term_kips"], 65.0, places=1)
        self.assertAlmostEqual(res_pass["rn_yielding_kips"], 200.0, places=1)
        self.assertAlmostEqual(res_pass["rn_rupture_kips"], 201.5, places=1)
        self.assertAlmostEqual(res_pass["phi_rn_kips"], 150.0, places=1)
        self.assertAlmostEqual(res_pass["dcr"], 80.0 / 150.0, places=4)

        # Passing case governed by shear rupture limit
        res_rupture = verify_aisc_block_shear(
            pu_kips=90.0,
            agv_in2=6.0,
            anv_in2=3.0,
            ant_in2=1.0,
            fy_ksi=50.0,
            fu_ksi=65.0,
            u_bs=1.0,
        )
        self.assertTrue(res_rupture["pass"])
        self.assertEqual(res_rupture["governing_limit_state"], "shear_rupture")
        self.assertAlmostEqual(res_rupture["rn_rupture_kips"], 182.0, places=1)
        self.assertAlmostEqual(res_rupture["phi_rn_kips"], 136.5, places=1)

        # Non-uniform tension stress reduction (Ubs = 0.5 for coped beam)
        res_coped = verify_aisc_block_shear(
            pu_kips=70.0,
            agv_in2=4.5,
            anv_in2=3.5,
            ant_in2=1.0,
            u_bs=0.5,
        )
        self.assertTrue(res_coped["pass"])
        self.assertAlmostEqual(res_coped["tension_term_kips"], 32.5, places=1)

        # Overloaded demand failure (dcr > 1.0)
        res_fail = verify_aisc_block_shear(
            pu_kips=160.0,
            agv_in2=4.5,
            anv_in2=3.5,
            ant_in2=1.0,
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr luxuries", "dcr"], 1.0) if "dcr luxuries" in res_fail else self.assertGreater(res_fail["dcr"], 1.0)

        # Geometry violation: net shear area exceeds gross shear area
        res_area_err = verify_aisc_block_shear(
            pu_kips=50.0,
            agv_in2=3.0,
            anv_in2=3.5,
            ant_in2=1.0,
        )
        self.assertFalse(res_area_err["pass"])
        self.assertEqual(res_area_err["reason"], "net_shear_area_exceeds_gross")

        # Net tension area exceeds provided gross tension area
        res_tens_err = verify_aisc_block_shear(
            pu_kips=50.0,
            agv_in2=4.5,
            anv_in2=3.5,
            ant_in2=1.5,
            agt_in2=1.2,
        )
        self.assertFalse(res_tens_err["pass"])
        self.assertEqual(res_tens_err["reason"], "net_tension_area_exceeds_gross")

        # Invalid shear lag factor Ubs
        res_ubs_err = verify_aisc_block_shear(
            pu_kips=50.0,
            agv_in2=4.5,
            anv_in2=3.5,
            ant_in2=1.0,
            u_bs=1.25,
        )
        self.assertFalse(res_ubs_err["pass"])
        self.assertEqual(res_ubs_err["reason"], "invalid_shear_lag_factor")

        # NaN and negative input validation
        res_nan = verify_aisc_block_shear(
            pu_kips=float("nan"),
            agv_in2=4.5,
            anv_in2=3.5,
            ant_in2=1.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_aisc_block_shear(
            pu_kips=-20.0,
            agv_in2=4.5,
            anv_in2=3.5,
            ant_in2=1.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_load_input")


    def test_asce7_snow_drift(self) -> None:
        from harness.verifiers import verify_asce7_snow_drift

        # Standard passing leeward drift check (hd <= hc)
        res_pass = verify_asce7_snow_drift(
            pg_psf=30.0,
            lu_fetch_ft=100.0,
            step_height_ft=10.0,
            p_capacity_psf=100.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertTrue(res_pass["drift_required"])
        self.assertAlmostEqual(res_pass["snow_density_pcf"], 17.9, places=1)
        self.assertAlmostEqual(res_pass["pf_lower_psf"], 21.0, places=1)
        self.assertAlmostEqual(res_pass["drift_height_ft"], 3.519, places=2)
        self.assertAlmostEqual(res_pass["drift_width_ft"], 14.08, places=1)
        self.assertAlmostEqual(res_pass["p_drift_psf"], 63.00, places=1)
        self.assertAlmostEqual(res_pass["p_total_psf"], 84.00, places=1)
        self.assertAlmostEqual(res_pass["dcr"], 0.840, places=3)

        # Truncated drift condition (hd > hc) with widened drift profile
        res_trunc = verify_asce7_snow_drift(
            pg_psf=30.0,
            lu_fetch_ft=100.0,
            step_height_ft=3.0,
            p_capacity_psf=70.0,
        )
        self.assertTrue(res_trunc["pass"])
        self.assertTrue(res_trunc["drift_required"])
        self.assertAlmostEqual(res_trunc["drift_height_ft"], res_trunc["clear_height_ft"], places=2)
        self.assertAlmostEqual(res_trunc["drift_width_ft"], 8.0 * res_trunc["clear_height_ft"], places=1)

        # Small step height where hc / hb < 0.2 exempts drift surcharge
        res_exempt = verify_asce7_snow_drift(
            pg_psf=30.0,
            lu_fetch_ft=100.0,
            step_height_ft=1.3,
            p_capacity_psf=30.0,
        )
        self.assertTrue(res_exempt["pass"])
        self.assertFalse(res_exempt["drift_required"])
        self.assertEqual(res_exempt["p_drift_psf"], 0.0)
        self.assertEqual(res_exempt["p_total_psf"], res_exempt["pf_lower_psf"])

        # Capacity exceeded failure
        res_fail = verify_asce7_snow_drift(
            pg_psf=30.0,
            lu_fetch_ft=100.0,
            step_height_ft=10.0,
            p_capacity_psf=70.0,
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr"], 1.0)

        # Windward drift check (3/4 leeward height)
        res_windward = verify_asce7_snow_drift(
            pg_psf=30.0,
            lu_fetch_ft=100.0,
            step_height_ft=10.0,
            p_capacity_psf=100.0,
            is_windward=True,
        )
        self.assertTrue(res_windward["pass"])
        self.assertAlmostEqual(res_windward["drift_height_ft"], 0.75 * res_pass["drift_height_ft"], places=2)

        # Input error validation checks
        res_nan = verify_asce7_snow_drift(
            pg_psf=float("nan"),
            lu_fetch_ft=100.0,
            step_height_ft=10.0,
            p_capacity_psf=80.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_asce7_snow_drift(
            pg_psf=-30.0,
            lu_fetch_ft=100.0,
            step_height_ft=10.0,
            p_capacity_psf=80.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_loading_or_geometry")


    def test_nrcs_tr55_runoff(self) -> None:
        from harness.verifiers import verify_nrcs_tr55_runoff

        # Standard passing 24-hr storm runoff check (P = 5.0 in, CN = 75)
        res_pass = verify_nrcs_tr55_runoff(
            rainfall_depth_in=5.0,
            curve_number_cn=75.0,
            drainage_area_acres=10.0,
            storage_capacity_acre_ft=2.5,
        )
        self.assertTrue(res_pass["pass"])
        self.assertTrue(res_pass["runoff_generated"])
        self.assertAlmostEqual(res_pass["potential_retention_s_in"], 3.3333, places=3)
        self.assertAlmostEqual(res_pass["initial_abstraction_ia_in"], 0.6667, places=3)
        self.assertAlmostEqual(res_pass["runoff_depth_in"], 2.4493, places=2)
        self.assertAlmostEqual(res_pass["runoff_volume_acre_ft"], 2.0411, places=2)
        self.assertAlmostEqual(res_pass["dcr"], 2.0411 / 2.5, places=3)
        self.assertLess(res_pass["dcr"], 1.0)

        # Low precipitation below initial abstraction (P <= Ia) produces zero runoff
        res_no_runoff = verify_nrcs_tr55_runoff(
            rainfall_depth_in=1.0,
            curve_number_cn=60.0,
            drainage_area_acres=10.0,
            storage_capacity_acre_ft=1.0,
        )
        self.assertTrue(res_no_runoff["pass"])
        self.assertFalse(res_no_runoff["runoff_generated"])
        self.assertEqual(res_no_runoff["runoff_depth_in"], 0.0)
        self.assertEqual(res_no_runoff["runoff_volume_acre_ft"], 0.0)
        self.assertEqual(res_no_runoff["dcr"], 0.0)

        # 100% impervious surface (CN = 100 -> S = 0, Q = P)
        res_impervious = verify_nrcs_tr55_runoff(
            rainfall_depth_in=4.0,
            curve_number_cn=100.0,
            drainage_area_acres=5.0,
            storage_capacity_acre_ft=2.0,
        )
        self.assertTrue(res_impervious["pass"])
        self.assertEqual(res_impervious["potential_retention_s_in"], 0.0)
        self.assertEqual(res_impervious["runoff_depth_in"], 4.0)
        self.assertAlmostEqual(res_impervious["runoff_volume_acre_ft"], (4.0 / 12.0) * 5.0, places=3)

        # Modern NRCS parameter revision (lambda = 0.05)
        res_lambda05 = verify_nrcs_tr55_runoff(
            rainfall_depth_in=5.0,
            curve_number_cn=75.0,
            drainage_area_acres=10.0,
            storage_capacity_acre_ft=3.0,
            initial_abstraction_ratio=0.05,
        )
        self.assertTrue(res_lambda05["pass"])
        self.assertGreater(res_lambda05["runoff_depth_in"], res_pass["runoff_depth_in"])

        # Storage volume exceeded failure (DCR > 1.0)
        res_fail_vol = verify_nrcs_tr55_runoff(
            rainfall_depth_in=6.0,
            curve_number_cn=85.0,
            drainage_area_acres=20.0,
            storage_capacity_acre_ft=4.0,
        )
        self.assertFalse(res_fail_vol["pass"])
        self.assertGreater(res_fail_vol["dcr"], 1.0)

        # Allowable runoff depth exceeded failure
        res_fail_depth = verify_nrcs_tr55_runoff(
            rainfall_depth_in=5.0,
            curve_number_cn=80.0,
            drainage_area_acres=10.0,
            storage_capacity_acre_ft=5.0,
            allowable_runoff_depth_in=2.0,
        )
        self.assertFalse(res_fail_depth["pass"])
        self.assertGreater(res_fail_depth["dcr"], 1.0)

        # Input error validation checks
        res_nan = verify_nrcs_tr55_runoff(
            rainfall_depth_in=float("nan"),
            curve_number_cn=75.0,
            drainage_area_acres=10.0,
            storage_capacity_acre_ft=2.5,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_nrcs_tr55_runoff(
            rainfall_depth_in=-2.0,
            curve_number_cn=75.0,
            drainage_area_acres=10.0,
            storage_capacity_acre_ft=2.5,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "negative_rainfall_depth")

        res_inv_cn = verify_nrcs_tr55_runoff(
            rainfall_depth_in=4.0,
            curve_number_cn=115.0,
            drainage_area_acres=10.0,
            storage_capacity_acre_ft=2.5,
        )
        self.assertFalse(res_inv_cn["pass"])
        self.assertEqual(res_inv_cn["reason"], "invalid_curve_number")


    def test_open_channel_hydraulic_jump(self) -> None:
        from harness.verifiers import verify_open_channel_hydraulic_jump

        # Standard passing steady hydraulic jump in spillway stilling basin
        res_pass = verify_open_channel_hydraulic_jump(
            discharge_cfs=360.0,
            channel_width_ft=10.0,
            upstream_depth_ft=1.0,
            provided_basin_length_ft=60.0,
            tailwater_depth_ft=8.5,
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["jump_classification"], "steady")
        self.assertAlmostEqual(res_pass["froude_upstream"], 6.347, places=2)
        self.assertAlmostEqual(res_pass["sequent_depth_ft"], 8.490, places=2)
        self.assertAlmostEqual(res_pass["required_basin_length_ft"], 50.94, places=1)
        self.assertAlmostEqual(res_pass["head_loss_ft"], 12.37, places=1)
        self.assertAlmostEqual(res_pass["dissipation_efficiency"], 0.585, places=2)
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertTrue(res_pass["length_ok"])
        self.assertTrue(res_pass["froude_ok"])
        self.assertTrue(res_pass["tailwater_ok"])

        # Undersized basin length failure
        res_fail_length = verify_open_channel_hydraulic_jump(
            discharge_cfs=360.0,
            channel_width_ft=10.0,
            upstream_depth_ft=1.0,
            provided_basin_length_ft=40.0,
        )
        self.assertFalse(res_fail_length["pass"])
        self.assertFalse(res_fail_length["length_ok"])
        self.assertGreater(res_fail_length["dcr"], 1.0)

        # Subcritical upstream flow (jump cannot form)
        res_subcritical = verify_open_channel_hydraulic_jump(
            discharge_cfs=50.0,
            channel_width_ft=10.0,
            upstream_depth_ft=3.0,
            provided_basin_length_ft=20.0,
        )
        self.assertFalse(res_subcritical["pass"])
        self.assertEqual(res_subcritical["reason"], "upstream_flow_subcritical")

        # Low Froude number (undular jump with inadequate dissipation)
        res_weak = verify_open_channel_hydraulic_jump(
            discharge_cfs=75.0,
            channel_width_ft=10.0,
            upstream_depth_ft=1.0,
            provided_basin_length_ft=20.0,
            min_froude_number=1.7,
        )
        self.assertFalse(res_weak["pass"])
        self.assertFalse(res_weak["froude_ok"])
        self.assertEqual(res_weak["jump_classification"], "undular")

        # Tailwater sweepout failure (tailwater < 0.95 * y2)
        res_sweepout = verify_open_channel_hydraulic_jump(
            discharge_cfs=360.0,
            channel_width_ft=10.0,
            upstream_depth_ft=1.0,
            provided_basin_length_ft=60.0,
            tailwater_depth_ft=6.0,
        )
        self.assertFalse(res_sweepout["pass"])
        self.assertFalse(res_sweepout["tailwater_ok"])

        # Input error validation checks
        res_nan = verify_open_channel_hydraulic_jump(
            discharge_cfs=float("nan"),
            channel_width_ft=10.0,
            upstream_depth_ft=1.0,
            provided_basin_length_ft=60.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_open_channel_hydraulic_jump(
            discharge_cfs=-100.0,
            channel_width_ft=10.0,
            upstream_depth_ft=1.0,
            provided_basin_length_ft=60.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_geometry_or_flow")


    def test_heat_exchanger_lmtd(self) -> None:
        from harness.verifiers import verify_heat_exchanger_lmtd

        # Standard passing counterflow water-to-water heat exchanger
        res_pass = verify_heat_exchanger_lmtd(
            q_design_btuh=400000.0,
            th_in_f=180.0,
            th_out_f=140.0,
            tc_in_f=60.0,
            tc_out_f=100.0,
            u_overall_btu_hr_sqft_f=100.0,
            area_sqft=60.0,
            flow_type="counterflow",
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["lmtd_f"], 80.0)
        self.assertEqual(res_pass["f_factor"], 1.0)
        self.assertEqual(res_pass["area_required_sqft"], 50.0)
        self.assertAlmostEqual(res_pass["dcr"], 400000.0 / 480000.0, places=4)

        # 1-2 shell and tube exchanger with F-factor correction
        res_st = verify_heat_exchanger_lmtd(
            q_design_btuh=500000.0,
            th_in_f=200.0,
            th_out_f=140.0,
            tc_in_f=80.0,
            tc_out_f=120.0,
            u_overall_btu_hr_sqft_f=150.0,
            area_sqft=60.0,
            flow_type="shell_tube_1_2",
        )
        self.assertTrue(res_st["pass"])
        self.assertAlmostEqual(res_st["f_factor"], 0.9105, places=2)
        self.assertLess(res_st["dcr"], 1.0)

        # Undersized surface area failure
        res_fail_area = verify_heat_exchanger_lmtd(
            q_design_btuh=600000.0,
            th_in_f=180.0,
            th_out_f=140.0,
            tc_in_f=60.0,
            tc_out_f=100.0,
            u_overall_btu_hr_sqft_f=100.0,
            area_sqft=50.0,
        )
        self.assertFalse(res_fail_area["pass"])
        self.assertGreater(res_fail_area["dcr"], 1.0)

        # Temperature cross in parallel flow
        res_cross_par = verify_heat_exchanger_lmtd(
            q_design_btuh=200000.0,
            th_in_f=150.0,
            th_out_f=100.0,
            tc_in_f=70.0,
            tc_out_f=110.0,
            u_overall_btu_hr_sqft_f=100.0,
            area_sqft=50.0,
            flow_type="parallel",
        )
        self.assertFalse(res_cross_par["pass"])
        self.assertEqual(res_cross_par["reason"], "parallel_flow_temperature_cross")

        # Temperature cross in counterflow
        res_cross_cnt = verify_heat_exchanger_lmtd(
            q_design_btuh=200000.0,
            th_in_f=150.0,
            th_out_f=100.0,
            tc_in_f=70.0,
            tc_out_f=160.0,
            u_overall_btu_hr_sqft_f=100.0,
            area_sqft=50.0,
            flow_type="counterflow",
        )
        self.assertFalse(res_cross_cnt["pass"])
        self.assertEqual(res_cross_cnt["reason"], "temperature_cross_violation")

        # Thermodynamic violation: hot stream heating instead of cooling
        res_heat_err = verify_heat_exchanger_lmtd(
            q_design_btuh=200000.0,
            th_in_f=140.0,
            th_out_f=160.0,
            tc_in_f=60.0,
            tc_out_f=100.0,
            u_overall_btu_hr_sqft_f=100.0,
            area_sqft=50.0,
        )
        self.assertFalse(res_heat_err["pass"])
        self.assertEqual(res_heat_err["reason"], "hot_stream_does_not_cool")

        # NaN input validation
        res_nan = verify_heat_exchanger_lmtd(
            q_design_btuh=float("nan"),
            th_in_f=180.0,
            th_out_f=140.0,
            tc_in_f=60.0,
            tc_out_f=100.0,
            u_overall_btu_hr_sqft_f=100.0,
            area_sqft=50.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        # Non-positive input validation
        res_neg = verify_heat_exchanger_lmtd(
            q_design_btuh=-100000.0,
            th_in_f=180.0,
            th_out_f=140.0,
            tc_in_f=60.0,
            tc_out_f=100.0,
            u_overall_btu_hr_sqft_f=100.0,
            area_sqft=50.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_duty_or_geometry")


    def test_asme_bpvc_vessel_thickness(self) -> None:
        from harness.verifiers import verify_asme_bpvc_vessel_thickness

        # Standard cylindrical pressure vessel shell passing check
        res_pass = verify_asme_bpvc_vessel_thickness(
            design_pressure_psi=250.0,
            inside_diameter_in=48.0,
            nominal_thickness_in=0.50,
            allowable_stress_psi=20000.0,
            joint_efficiency_e=1.0,
            corrosion_allowance_in=0.0625,
            component_type="cylindrical_shell",
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["component_type"], "cylindrical_shell")
        self.assertAlmostEqual(res_pass["t_pressure_in"], 0.3023, places=3)
        self.assertAlmostEqual(res_pass["t_required_in"], 0.3648, places=3)
        self.assertAlmostEqual(res_pass["mawp_psi"], 360.64, places=1)
        self.assertLess(res_pass["dcr"], 1.0)

        # 2:1 Ellipsoidal head check
        res_ellip = verify_asme_bpvc_vessel_thickness(
            design_pressure_psi=250.0,
            inside_diameter_in=48.0,
            nominal_thickness_in=0.50,
            allowable_stress_psi=20000.0,
            joint_efficiency_e=1.0,
            corrosion_allowance_in=0.0625,
            component_type="2_to_1_ellipsoidal",
        )
        self.assertTrue(res_ellip["pass"])
        self.assertAlmostEqual(res_ellip["t_pressure_in"], 0.3004, places=3)
        self.assertAlmostEqual(res_ellip["mawp_psi"], 363.92, places=1)

        # Hemispherical head check (half the thickness of cylindrical shell)
        res_hemi = verify_asme_bpvc_vessel_thickness(
            design_pressure_psi=250.0,
            inside_diameter_in=48.0,
            nominal_thickness_in=0.375,
            allowable_stress_psi=20000.0,
            joint_efficiency_e=1.0,
            corrosion_allowance_in=0.0625,
            component_type="hemispherical",
        )
        self.assertTrue(res_hemi["pass"])
        self.assertAlmostEqual(res_hemi["t_pressure_in"], 0.1502, places=3)
        self.assertGreater(res_hemi["mawp_psi"], 500.0)

        # Torispherical head failing due to higher knuckle stress
        res_tori_fail = verify_asme_bpvc_vessel_thickness(
            design_pressure_psi=250.0,
            inside_diameter_in=48.0,
            nominal_thickness_in=0.50,
            allowable_stress_psi=20000.0,
            joint_efficiency_e=1.0,
            corrosion_allowance_in=0.0625,
            component_type="torispherical",
        )
        self.assertFalse(res_tori_fail["pass"])
        self.assertGreater(res_tori_fail["dcr"], 1.0)

        # Corrosion allowance consuming entire wall thickness
        res_corr_err = verify_asme_bpvc_vessel_thickness(
            design_pressure_psi=100.0,
            inside_diameter_in=24.0,
            nominal_thickness_in=0.25,
            corrosion_allowance_in=0.30,
        )
        self.assertFalse(res_corr_err["pass"])
        self.assertEqual(res_corr_err["reason"], "corrosion_allowance_exceeds_nominal_thickness")

        # Input error validation
        res_nan = verify_asme_bpvc_vessel_thickness(
            design_pressure_psi=float("nan"),
            inside_diameter_in=48.0,
            nominal_thickness_in=0.50,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_asme_bpvc_vessel_thickness(
            design_pressure_psi=-200.0,
            inside_diameter_in=48.0,
            nominal_thickness_in=0.50,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_pressure_or_geometry")

        res_inv_type = verify_asme_bpvc_vessel_thickness(
            design_pressure_psi=200.0,
            inside_diameter_in=48.0,
            nominal_thickness_in=0.50,
            component_type="conical_flange",
        )
        self.assertFalse(res_inv_type["pass"])
        self.assertEqual(res_inv_type["reason"], "invalid_component_type")


    def test_infinite_slope_stability(self) -> None:
        from harness.verifiers import verify_infinite_slope_stability

        # Standard dry cohesionless sand slope passing (FS = tan(phi) / tan(beta))
        res_pass = verify_infinite_slope_stability(
            slope_angle_deg=25.0,
            depth_z_ft=10.0,
            soil_cohesion_psf=0.0,
            soil_friction_angle_deg=35.0,
            soil_unit_weight_pcf=120.0,
            min_fs=1.3,
        )
        self.assertTrue(res_pass["pass"])
        self.assertAlmostEqual(res_pass["factor_of_safety"], 1.5016, places=3)
        self.assertAlmostEqual(res_pass["dcr"], 1.3 / 1.5016, places=3)
        self.assertEqual(res_pass["pore_pressure_psf"], 0.0)

        # Steady seepage parallel to slope triggering slope failure
        res_seepage_fail = verify_infinite_slope_stability(
            slope_angle_deg=25.0,
            depth_z_ft=10.0,
            soil_cohesion_psf=0.0,
            soil_friction_angle_deg=35.0,
            soil_unit_weight_pcf=120.0,
            water_table_height_ft=10.0,
            sat_unit_weight_pcf=125.0,
            min_fs=1.3,
        )
        self.assertFalse(res_seepage_fail["pass"])
        self.assertLess(res_seepage_fail["factor_of_safety"], 1.0)
        self.assertGreater(res_seepage_fail["dcr"], 1.0)
        self.assertGreater(res_seepage_fail["pore_pressure_psf"], 500.0)

        # Cohesion restoration stabilizing the saturated slope
        res_cohesive_pass = verify_infinite_slope_stability(
            slope_angle_deg=25.0,
            depth_z_ft=10.0,
            soil_cohesion_psf=300.0,
            soil_friction_angle_deg=35.0,
            soil_unit_weight_pcf=120.0,
            water_table_height_ft=10.0,
            sat_unit_weight_pcf=125.0,
            min_fs=1.3,
        )
        self.assertTrue(res_cohesive_pass["pass"])
        self.assertGreaterEqual(res_cohesive_pass["factor_of_safety"], 1.3)

        # Pseudostatic earthquake coefficient reducing stability
        res_seismic = verify_infinite_slope_stability(
            slope_angle_deg=25.0,
            depth_z_ft=10.0,
            soil_cohesion_psf=0.0,
            soil_friction_angle_deg=35.0,
            soil_unit_weight_pcf=120.0,
            seismic_coeff_kh=0.15,
            min_fs=1.3,
        )
        self.assertFalse(res_seismic["pass"])
        self.assertLess(res_seismic["factor_of_safety"], res_pass["factor_of_safety"])

        # Water table exceeding failure depth error
        res_hw_err = verify_infinite_slope_stability(
            slope_angle_deg=25.0,
            depth_z_ft=10.0,
            soil_cohesion_psf=0.0,
            soil_friction_angle_deg=35.0,
            soil_unit_weight_pcf=120.0,
            water_table_height_ft=15.0,
        )
        self.assertFalse(res_hw_err["pass"])
        self.assertEqual(res_hw_err["reason"], "water_table_exceeds_depth")

        # Invalid slope angle error
        res_angle_err = verify_infinite_slope_stability(
            slope_angle_deg=92.0,
            depth_z_ft=10.0,
            soil_cohesion_psf=0.0,
            soil_friction_angle_deg=35.0,
            soil_unit_weight_pcf=120.0,
        )
        self.assertFalse(res_angle_err["pass"])
        self.assertEqual(res_angle_err["reason"], "invalid_slope_angle")

        # NaN input validation
        res_nan = verify_infinite_slope_stability(
            slope_angle_deg=float("nan"),
            depth_z_ft=10.0,
            soil_cohesion_psf=0.0,
            soil_friction_angle_deg=35.0,
            soil_unit_weight_pcf=120.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")


    def test_water_hammer_surge(self) -> None:
        from harness.verifiers import verify_water_hammer_surge

        # Standard passing rapid closure in PVC pipeline (AWWA C900)
        res_pass = verify_water_hammer_surge(
            pipe_inner_diameter_in=6.0,
            wall_thickness_in=0.30,
            pipe_length_ft=2000.0,
            valve_closure_time_s=1.0,
            static_pressure_psi=60.0,
            pipe_pressure_rating_psi=150.0,
            flow_velocity_fps=5.0,
            pipe_modulus_psi=400000.0,
            surge_allowance_pct=50.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["closure_regime"], "rapid")
        self.assertAlmostEqual(res_pass["wave_speed_fps"], 1179.88, places=1)
        self.assertAlmostEqual(res_pass["critical_time_s"], 3.3902, places=2)
        self.assertAlmostEqual(res_pass["surge_pressure_psi"], 79.46, places=1)
        self.assertAlmostEqual(res_pass["total_pressure_psi"], 139.46, places=1)
        self.assertEqual(res_pass["allowable_pressure_psi"], 225.0)
        self.assertLess(res_pass["dcr"], 1.0)

        # Slow closure condition with attenuated surge pressure (Tc < tv)
        res_slow = verify_water_hammer_surge(
            pipe_inner_diameter_in=6.0,
            wall_thickness_in=0.30,
            pipe_length_ft=2000.0,
            valve_closure_time_s=10.0,
            static_pressure_psi=60.0,
            pipe_pressure_rating_psi=150.0,
            flow_velocity_fps=5.0,
            pipe_modulus_psi=400000.0,
        )
        self.assertTrue(res_slow["pass"])
        self.assertEqual(res_slow["closure_regime"], "slow")
        self.assertAlmostEqual(res_slow["surge_pressure_psi"], 26.94, places=1)
        self.assertLess(res_slow["total_pressure_psi"], res_pass["total_pressure_psi"])

        # Flow specified via flow rate GPM
        res_gpm = verify_water_hammer_surge(
            pipe_inner_diameter_in=6.0,
            wall_thickness_in=0.30,
            pipe_length_ft=2000.0,
            valve_closure_time_s=1.0,
            static_pressure_psi=60.0,
            pipe_pressure_rating_psi=150.0,
            flow_rate_gpm=440.0,
        )
        self.assertTrue(res_gpm["pass"])
        self.assertAlmostEqual(res_gpm["flow_velocity_fps"], 4.99, places=1)

        # Capacity exceeded failure (high static pressure with no surge allowance)
        res_fail = verify_water_hammer_surge(
            pipe_inner_diameter_in=6.0,
            wall_thickness_in=0.30,
            pipe_length_ft=2000.0,
            valve_closure_time_s=1.0,
            static_pressure_psi=100.0,
            pipe_pressure_rating_psi=150.0,
            flow_velocity_fps=5.0,
            surge_allowance_pct=0.0,
        )
        self.assertFalse(res_fail["pass"])
        self.assertGreater(res_fail["dcr"], 1.0)

        # Input error validation checks
        res_nan = verify_water_hammer_surge(
            pipe_inner_diameter_in=float("nan"),
            wall_thickness_in=0.30,
            pipe_length_ft=2000.0,
            valve_closure_time_s=1.0,
            static_pressure_psi=60.0,
            pipe_pressure_rating_psi=150.0,
            flow_velocity_fps=5.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_water_hammer_surge(
            pipe_inner_diameter_in=6.0,
            wall_thickness_in=0.30,
            pipe_length_ft=2000.0,
            valve_closure_time_s=-1.0,
            static_pressure_psi=60.0,
            pipe_pressure_rating_psi=150.0,
            flow_velocity_fps=5.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_closure_time")

        res_no_flow = verify_water_hammer_surge(
            pipe_inner_diameter_in=6.0,
            wall_thickness_in=0.30,
            pipe_length_ft=2000.0,
            valve_closure_time_s=1.0,
            static_pressure_psi=60.0,
            pipe_pressure_rating_psi=150.0,
        )
        self.assertFalse(res_no_flow["pass"])
        self.assertEqual(res_no_flow["reason"], "zero_flow_input")


    def test_weir_discharge_capacity(self) -> None:
        from harness.verifiers import verify_weir_discharge_capacity

        # Standard suppressed rectangular weir passing
        res_supp = verify_weir_discharge_capacity(
            head_ft=1.5,
            crest_length_ft=10.0,
            weir_type="suppressed_rectangular",
            required_discharge_cfs=50.0,
            freeboard_ft=1.5,
            min_freeboard_ft=1.0,
        )
        self.assertTrue(res_supp["pass"])
        self.assertAlmostEqual(res_supp["q_capacity_cfs"], 61.176, places=1)
        self.assertAlmostEqual(res_supp["dcr"], 50.0 / 61.176, places=3)
        self.assertFalse(res_supp["is_submerged"])
        self.assertTrue(res_supp["freeboard_ok"])

        # Contracted rectangular weir with end contraction deduction
        res_cont = verify_weir_discharge_capacity(
            head_ft=1.0,
            crest_length_ft=10.0,
            weir_type="contracted_rectangular",
            required_discharge_cfs=30.0,
        )
        self.assertTrue(res_cont["pass"])
        self.assertEqual(res_cont["effective_length_ft"], 9.8)
        self.assertAlmostEqual(res_cont["q_capacity_cfs"], 3.33 * 9.8, places=2)

        # 90-degree V-notch weir
        res_vnotch = verify_weir_discharge_capacity(
            head_ft=1.2,
            weir_type="v_notch",
            notch_angle_deg=90.0,
            required_discharge_cfs=3.5,
        )
        self.assertTrue(res_vnotch["pass"])
        self.assertAlmostEqual(res_vnotch["q_capacity_cfs"], 3.948, places=1)
        self.assertLess(res_vnotch["dcr"], 1.0)

        # Cipolletti weir check
        res_cipo = verify_weir_discharge_capacity(
            head_ft=1.0,
            crest_length_ft=6.0,
            weir_type="cipolletti",
            required_discharge_cfs=18.0,
        )
        self.assertTrue(res_cipo["pass"])
        self.assertAlmostEqual(res_cipo["q_capacity_cfs"], 20.202, places=2)

        # Broad-crested weir check
        res_bcw = verify_weir_discharge_capacity(
            head_ft=1.0,
            crest_length_ft=8.0,
            weir_type="broad_crested",
            required_discharge_cfs=20.0,
        )
        self.assertTrue(res_bcw["pass"])
        self.assertAlmostEqual(res_bcw["q_capacity_cfs"], 24.698, places=1)

        # Villemonte submerged flow check
        res_sub = verify_weir_discharge_capacity(
            head_ft=2.0,
            crest_length_ft=10.0,
            tailwater_head_ft=1.0,
            required_discharge_cfs=75.0,
        )
        self.assertTrue(res_sub["pass"])
        self.assertTrue(res_sub["is_submerged"])
        self.assertAlmostEqual(res_sub["submergence_factor"], 0.8458, places=3)
        self.assertLess(res_sub["q_capacity_cfs"], res_sub["q_unsubmerged_cfs"])

        # Demand exceeding capacity failure
        res_fail_cap = verify_weir_discharge_capacity(
            head_ft=1.0,
            crest_length_ft=5.0,
            required_discharge_cfs=25.0,
        )
        self.assertFalse(res_fail_cap["pass"])
        self.assertGreater(res_fail_cap["dcr"], 1.0)

        # Freeboard violation failure
        res_fail_fb = verify_weir_discharge_capacity(
            head_ft=1.0,
            crest_length_ft=10.0,
            required_discharge_cfs=20.0,
            freeboard_ft=0.5,
            min_freeboard_ft=1.0,
        )
        self.assertFalse(res_fail_fb["pass"])
        self.assertFalse(res_fail_fb["freeboard_ok"])

        # Tailwater drowning weir error
        res_drown = verify_weir_discharge_capacity(
            head_ft=1.0,
            crest_length_ft=10.0,
            tailwater_head_ft=1.5,
        )
        self.assertFalse(res_drown["pass"])
        self.assertEqual(res_drown["reason"], "tailwater_drowns_weir")

        # Input error validation
        res_nan = verify_weir_discharge_capacity(
            head_ft=float("nan"),
            crest_length_ft=10.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg_head = verify_weir_discharge_capacity(
            head_ft=-1.0,
            crest_length_ft=10.0,
        )
        self.assertFalse(res_neg_head["pass"])
        self.assertEqual(res_neg_head["reason"], "non_positive_head")

        res_inv_type = verify_weir_discharge_capacity(
            head_ft=1.0,
            crest_length_ft=10.0,
            weir_type="parshall_flume",
        )
        self.assertFalse(res_inv_type["pass"])
        self.assertEqual(res_inv_type["reason"], "invalid_weir_type")


    def test_open_channel_trapezoidal_flow(self) -> None:
        from harness.verifiers import verify_open_channel_trapezoidal_flow

        # Standard passing subcritical trapezoidal channel (2H:1V side slopes)
        res_pass = verify_open_channel_trapezoidal_flow(
            q_design_cfs=45.0,
            bottom_width_ft=6.0,
            flow_depth_ft=2.0,
            channel_slope_ft_per_ft=0.002,
            side_slope_z=2.0,
            manning_n=0.030,
            channel_total_depth_ft=3.5,
            min_freeboard_ft=1.0,
            permissible_shear_psf=0.50,
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["flow_regime"], "subcritical")
        self.assertEqual(res_pass["flow_area_sqft"], 20.0)
        self.assertAlmostEqual(res_pass["hydraulic_radius_ft"], 1.338, places=3)
        self.assertAlmostEqual(res_pass["v_normal_fps"], 2.690, places=2)
        self.assertAlmostEqual(res_pass["q_capacity_cfs"], 53.80, places=1)
        self.assertAlmostEqual(res_pass["froude_number"], 0.397, places=2)
        self.assertAlmostEqual(res_pass["shear_stress_psf"], 0.2496, places=3)
        self.assertEqual(res_pass["freeboard_ft"], 1.5)
        self.assertLess(res_pass["dcr"], 1.0)

        # Rectangular channel passing (side_slope_z = 0)
        res_rect = verify_open_channel_trapezoidal_flow(
            q_design_cfs=30.0,
            bottom_width_ft=5.0,
            flow_depth_ft=2.0,
            channel_slope_ft_per_ft=0.003,
            side_slope_z=0.0,
            manning_n=0.020,
        )
        self.assertTrue(res_rect["pass"])
        self.assertEqual(res_rect["flow_area_sqft"], 10.0)
        self.assertGreater(res_rect["q_capacity_cfs"], 30.0)

        # Supercritical steep slope regime check
        res_super = verify_open_channel_trapezoidal_flow(
            q_design_cfs=80.0,
            bottom_width_ft=4.0,
            flow_depth_ft=1.5,
            channel_slope_ft_per_ft=0.035,
            side_slope_z=1.5,
            manning_n=0.025,
            max_velocity_fps=15.0,
        )
        self.assertTrue(res_super["pass"])
        self.assertEqual(res_super["flow_regime"], "supercritical")
        self.assertGreater(res_super["froude_number"], 1.0)

        # Discharge demand exceeding channel capacity failure
        res_fail_cap = verify_open_channel_trapezoidal_flow(
            q_design_cfs=70.0,
            bottom_width_ft=6.0,
            flow_depth_ft=2.0,
            channel_slope_ft_per_ft=0.002,
            side_slope_z=2.0,
            manning_n=0.030,
        )
        self.assertFalse(res_fail_cap["pass"])
        self.assertFalse(res_fail_cap["capacity_ok"])
        self.assertGreater(res_fail_cap["dcr"], 1.0)

        # Freeboard violation failure
        res_fail_fb = verify_open_channel_trapezoidal_flow(
            q_design_cfs=40.0,
            bottom_width_ft=6.0,
            flow_depth_ft=2.0,
            channel_slope_ft_per_ft=0.002,
            side_slope_z=2.0,
            channel_total_depth_ft=2.5,
            min_freeboard_ft=1.0,
        )
        self.assertFalse(res_fail_fb["pass"])
        self.assertFalse(res_fail_fb["freeboard_ok"])
        self.assertEqual(res_fail_fb["freeboard_ft"], 0.5)

        # Bed shear stress exceeding permissible lining threshold
        res_fail_shear = verify_open_channel_trapezoidal_flow(
            q_design_cfs=45.0,
            bottom_width_ft=6.0,
            flow_depth_ft=2.0,
            channel_slope_ft_per_ft=0.005,
            side_slope_z=2.0,
            permissible_shear_psf=0.40,
        )
        self.assertFalse(res_fail_shear["pass"])
        self.assertFalse(res_fail_shear["shear_ok"])
        self.assertGreater(res_fail_shear["shear_stress_psf"], 0.40)

        # Flow depth exceeding total channel depth error
        res_overflow = verify_open_channel_trapezoidal_flow(
            q_design_cfs=40.0,
            bottom_width_ft=6.0,
            flow_depth_ft=3.0,
            channel_slope_ft_per_ft=0.002,
            channel_total_depth_ft=2.5,
        )
        self.assertFalse(res_overflow["pass"])
        self.assertEqual(res_overflow["reason"], "flow_depth_exceeds_channel_depth")

        # Input error validation
        res_nan = verify_open_channel_trapezoidal_flow(
            q_design_cfs=float("nan"),
            bottom_width_ft=6.0,
            flow_depth_ft=2.0,
            channel_slope_ft_per_ft=0.002,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_open_channel_trapezoidal_flow(
            q_design_cfs=40.0,
            bottom_width_ft=-6.0,
            flow_depth_ft=2.0,
            channel_slope_ft_per_ft=0.002,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_geometry_or_channel_slope")

        res_inv_limits = verify_open_channel_trapezoidal_flow(
            q_design_cfs=40.0,
            bottom_width_ft=6.0,
            flow_depth_ft=2.0,
            channel_slope_ft_per_ft=0.002,
            min_velocity_fps=8.0,
            max_velocity_fps=4.0,
        )
        self.assertFalse(res_inv_limits["pass"])
        self.assertEqual(res_inv_limits["reason"], "invalid_velocity_limits")


    def test_nfpa92_stairwell_pressurization(self) -> None:
        from harness.verifiers import verify_nfpa92_stairwell_pressurization

        # Standard passing stairwell pressurization check in sprinklered high-rise
        res_pass = verify_nfpa92_stairwell_pressurization(
            design_pressure_diff_in_wg=0.15,
            door_width_in=36.0,
            door_height_in=84.0,
            door_closer_force_lbf=12.0,
            fan_airflow_cfm=14000.0,
            num_open_doors=2,
            num_closed_doors=10,
            open_door_velocity_fpm=200.0,
            leakage_area_per_door_sqft=0.25,
        )
        self.assertTrue(res_pass["pass"])
        self.assertAlmostEqual(res_pass["total_door_force_lbf"], 20.94, places=1)
        self.assertAlmostEqual(res_pass["max_pressure_diff_in_wg"], 0.3019, places=2)
        self.assertAlmostEqual(res_pass["q_open_doors_cfm"], 8400.0, places=1)
        self.assertAlmostEqual(res_pass["q_leakage_doors_cfm"], 2527.13, places=1)
        self.assertAlmostEqual(res_pass["q_required_cfm"], 12566.2, places=1)
        self.assertLess(res_pass["dcr"], 1.0)
        self.assertTrue(res_pass["door_force_ok"])
        self.assertTrue(res_pass["pressure_ok"])

        # Overpressurization causing door opening force to exceed 30 lbf (IBC 1010.1.3 violation)
        res_overpress = verify_nfpa92_stairwell_pressurization(
            design_pressure_diff_in_wg=0.35,
            door_width_in=36.0,
            door_height_in=84.0,
            door_closer_force_lbf=12.0,
            fan_airflow_cfm=20000.0,
        )
        self.assertFalse(res_overpress["pass"])
        self.assertFalse(res_overpress["door_force_ok"])
        self.assertFalse(res_overpress["pressure_ok"])
        self.assertGreater(res_overpress["total_door_force_lbf"], 30.0)

        # Underpressurization failing minimum pressure difference boundary (dP < 0.10 in. wg)
        res_underpress = verify_nfpa92_stairwell_pressurization(
            design_pressure_diff_in_wg=0.06,
            door_width_in=36.0,
            door_height_in=84.0,
            door_closer_force_lbf=10.0,
            fan_airflow_cfm=14000.0,
            min_pressure_diff_in_wg=0.10,
        )
        self.assertFalse(res_underpress["pass"])
        self.assertFalse(res_underpress["pressure_ok"])

        # Undersized fan airflow capacity failure
        res_fan_fail = verify_nfpa92_stairwell_pressurization(
            design_pressure_diff_in_wg=0.15,
            door_width_in=36.0,
            door_height_in=84.0,
            door_closer_force_lbf=12.0,
            fan_airflow_cfm=10000.0,
            num_open_doors=2,
            num_closed_doors=10,
        )
        self.assertFalse(res_fan_fail["pass"])
        self.assertGreater(res_fan_fail["airflow_dcr"], 1.0)

        # Door closer force alone exceeds allowable door opening force
        res_closer_err = verify_nfpa92_stairwell_pressurization(
            design_pressure_diff_in_wg=0.15,
            door_width_in=36.0,
            door_height_in=84.0,
            door_closer_force_lbf=32.0,
            fan_airflow_cfm=15000.0,
            max_door_opening_force_lbf=30.0,
        )
        self.assertFalse(res_closer_err["pass"])
        self.assertEqual(res_closer_err["reason"], "closer_force_exceeds_allowable_opening_force")

        # Input error validation checks
        res_nan = verify_nfpa92_stairwell_pressurization(
            design_pressure_diff_in_wg=float("nan"),
            door_width_in=36.0,
            door_height_in=84.0,
            door_closer_force_lbf=12.0,
            fan_airflow_cfm=14000.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_nfpa92_stairwell_pressurization(
            design_pressure_diff_in_wg=-0.15,
            door_width_in=36.0,
            door_height_in=84.0,
            door_closer_force_lbf=12.0,
            fan_airflow_cfm=14000.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_geometry_or_pressure")

        res_setback = verify_nfpa92_stairwell_pressurization(
            design_pressure_diff_in_wg=0.15,
            door_width_in=36.0,
            door_height_in=84.0,
            door_closer_force_lbf=12.0,
            fan_airflow_cfm=14000.0,
            latch_setback_in=40.0,
        )
        self.assertFalse(res_setback["pass"])
        self.assertEqual(res_setback["reason"], "latch_setback_exceeds_door_width")


    def test_cooling_tower_performance(self) -> None:
        from harness.verifiers import verify_cooling_tower_performance

        # Standard passing commercial cooling tower check (900 gpm, 95F in / 85F out, 78F wet-bulb)
        res_pass = verify_cooling_tower_performance(
            cooling_water_flow_gpm=900.0,
            t_hot_water_in_f=95.0,
            t_cold_water_out_f=85.0,
            wet_bulb_temp_f=78.0,
            tower_capacity_tons=350.0,
            cycles_of_concentration=4.0,
            min_approach_f=5.0,
            max_makeup_gpm=12.0,
        )
        self.assertTrue(res_pass["pass"])
        self.assertEqual(res_pass["range_f"], 10.0)
        self.assertEqual(res_pass["approach_f"], 7.0)
        self.assertEqual(res_pass["q_rejected_btuh"], 4500000.0)
        self.assertEqual(res_pass["tons_required"], 300.0)
        self.assertAlmostEqual(res_pass["dcr"], 300.0 / 350.0, places=3)
        self.assertAlmostEqual(res_pass["evaporation_gpm"], 7.2, places=2)
        self.assertAlmostEqual(res_pass["drift_gpm"], 0.045, places=3)
        self.assertAlmostEqual(res_pass["blowdown_gpm"], 2.355, places=3)
        self.assertAlmostEqual(res_pass["makeup_gpm"], 9.6, places=2)
        self.assertTrue(res_pass["approach_ok"])
        self.assertTrue(res_pass["thermal_ok"])
        self.assertTrue(res_pass["makeup_ok"])

        # Undersized tower thermal capacity failure (DCR > 1.0)
        res_fail_cap = verify_cooling_tower_performance(
            cooling_water_flow_gpm=900.0,
            t_hot_water_in_f=95.0,
            t_cold_water_out_f=85.0,
            wet_bulb_temp_f=78.0,
            tower_capacity_tons=250.0,
        )
        self.assertFalse(res_fail_cap["pass"])
        self.assertFalse(res_fail_cap["thermal_ok"])
        self.assertGreater(res_fail_cap["dcr"], 1.0)

        # Approach violation (approach < 5.0 F thermodynamic threshold)
        res_approach_fail = verify_cooling_tower_performance(
            cooling_water_flow_gpm=900.0,
            t_hot_water_in_f=95.0,
            t_cold_water_out_f=81.0,
            wet_bulb_temp_f=78.0,
            tower_capacity_tons=500.0,
            min_approach_f=5.0,
        )
        self.assertFalse(res_approach_fail["pass"])
        self.assertFalse(res_approach_fail["approach_ok"])
        self.assertEqual(res_approach_fail["approach_f"], 3.0)

        # Makeup water flow capacity exceeded
        res_makeup_fail = verify_cooling_tower_performance(
            cooling_water_flow_gpm=900.0,
            t_hot_water_in_f=95.0,
            t_cold_water_out_f=85.0,
            wet_bulb_temp_f=78.0,
            tower_capacity_tons=350.0,
            max_makeup_gpm=8.0,
        )
        self.assertFalse(res_makeup_fail["pass"])
        self.assertFalse(res_makeup_fail["makeup_ok"])
        self.assertGreater(res_makeup_fail["dcr"], 1.0)

        # Thermodynamic violation: cold water at or below ambient wet bulb
        res_2nd_law = verify_cooling_tower_performance(
            cooling_water_flow_gpm=900.0,
            t_hot_water_in_f=95.0,
            t_cold_water_out_f=75.0,
            wet_bulb_temp_f=78.0,
            tower_capacity_tons=350.0,
        )
        self.assertFalse(res_2nd_law["pass"])
        self.assertEqual(res_2nd_law["reason"], "cold_water_below_wet_bulb")

        # Thermal violation: hot water not cooled
        res_heat_err = verify_cooling_tower_performance(
            cooling_water_flow_gpm=900.0,
            t_hot_water_in_f=85.0,
            t_cold_water_out_f=90.0,
            wet_bulb_temp_f=75.0,
            tower_capacity_tons=350.0,
        )
        self.assertFalse(res_heat_err["pass"])
        self.assertEqual(res_heat_err["reason"], "hot_water_not_cooled")

        # Invalid cycles of concentration (COC <= 1.0)
        res_coc_err = verify_cooling_tower_performance(
            cooling_water_flow_gpm=900.0,
            t_hot_water_in_f=95.0,
            t_cold_water_out_f=85.0,
            wet_bulb_temp_f=78.0,
            tower_capacity_tons=350.0,
            cycles_of_concentration=1.0,
        )
        self.assertFalse(res_coc_err["pass"])
        self.assertEqual(res_coc_err["reason"], "invalid_cycles_of_concentration")

        # Input error validation checks
        res_nan = verify_cooling_tower_performance(
            cooling_water_flow_gpm=float("nan"),
            t_hot_water_in_f=95.0,
            t_cold_water_out_f=85.0,
            wet_bulb_temp_f=78.0,
            tower_capacity_tons=350.0,
        )
        self.assertFalse(res_nan["pass"])
        self.assertEqual(res_nan["reason"], "nan_input")

        res_neg = verify_cooling_tower_performance(
            cooling_water_flow_gpm=-900.0,
            t_hot_water_in_f=95.0,
            t_cold_water_out_f=85.0,
            wet_bulb_temp_f=78.0,
            tower_capacity_tons=350.0,
        )
        self.assertFalse(res_neg["pass"])
        self.assertEqual(res_neg["reason"], "non_positive_flow_or_capacity")


if __name__ == "__main__":
    unittest.main()
