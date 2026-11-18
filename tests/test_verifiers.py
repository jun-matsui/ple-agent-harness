"""Unit tests for deterministic engineering code-gate verifiers."""

import unittest

from harness.verifiers import (
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


if __name__ == "__main__":
    unittest.main()
