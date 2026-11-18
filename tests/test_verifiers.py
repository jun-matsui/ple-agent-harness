"""Unit tests for deterministic engineering code-gate verifiers."""

import unittest

from harness.verifiers import (
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


if __name__ == "__main__":
    unittest.main()
