proc start_step { step } {
  set stopFile ".stop.rst"
  if {[file isfile .stop.rst]} {
    puts ""
    puts "*** Halting run - EA reset detected ***"
    puts ""
    puts ""
    return -code error
  }
  set beginFile ".$step.begin.rst"
  set platform "$::tcl_platform(platform)"
  set user "$::tcl_platform(user)"
  set pid [pid]
  set host ""
  if { [string equal $platform unix] } {
    if { [info exist ::env(HOSTNAME)] } {
      set host $::env(HOSTNAME)
    }
  } else {
    if { [info exist ::env(COMPUTERNAME)] } {
      set host $::env(COMPUTERNAME)
    }
  }
  set ch [open $beginFile w]
  puts $ch "<?xml version=\"1.0\"?>"
  puts $ch "<ProcessHandle Version=\"1\" Minor=\"0\">"
  puts $ch "    <Process Command=\".planAhead.\" Owner=\"$user\" Host=\"$host\" Pid=\"$pid\">"
  puts $ch "    </Process>"
  puts $ch "</ProcessHandle>"
  close $ch
}

proc end_step { step } {
  set endFile ".$step.end.rst"
  set ch [open $endFile w]
  close $ch
}

proc step_failed { step } {
  set endFile ".$step.error.rst"
  set ch [open $endFile w]
  close $ch
}

set_msg_config -id {HDL 9-1061} -limit 100000
set_msg_config -id {HDL 9-1654} -limit 100000

start_step place_design
set rc [catch {
  set_clock_uncertainty -from userclk2 -to userclk2 0.3 -setup
  create_msg_db place_design.pb
  set_param gui.test TreeTableDev
  read_checkpoint riffa_top_pcie_7x_v2_1_opt.dcp
  place_design 
  catch { report_io -file riffa_top_pcie_7x_v2_1_io_placed.rpt }
  catch { report_clock_utilization -file riffa_top_pcie_7x_v2_1_clock_utilization_placed.rpt }
  catch { report_utilization -file riffa_top_pcie_7x_v2_1_utilization_placed.rpt -pb riffa_top_pcie_7x_v2_1_utilization_placed.pb }
  catch { report_control_sets -verbose -file riffa_top_pcie_7x_v2_1_control_sets_placed.rpt }
  write_checkpoint -force riffa_top_pcie_7x_v2_1_placed.dcp
  close_msg_db -file place_design.pb
} RESULT]
if {$rc} {
  step_failed place_design
  return -code error $RESULT
} else {
  end_step place_design
}

start_step phys_opt_design
set rc [catch {
  create_msg_db phys_opt_design.pb
  phys_opt_design -directive Explore
  write_checkpoint -force riffa_top_pcie_7x_v2_1_physopt.dcp
  close_msg_db -file phys_opt_design.pb
} RESULT]
if {$rc} {
  step_failed phys_opt_design
  return -code error $RESULT
} else {
  end_step phys_opt_design
}

start_step route_design
set rc [catch {
  create_msg_db route_design.pb
  set_clock_uncertainty -from userclk2 -to userclk2 0 -setup
  route_design 
  catch { report_drc -file riffa_top_pcie_7x_v2_1_drc_routed.rpt -pb riffa_top_pcie_7x_v2_1_drc_routed.pb }
  catch { report_power -file riffa_top_pcie_7x_v2_1_power_routed.rpt -pb riffa_top_pcie_7x_v2_1_power_summary_routed.pb }
  catch { report_route_status -file riffa_top_pcie_7x_v2_1_route_status.rpt -pb riffa_top_pcie_7x_v2_1_route_status.pb }
  catch { report_timing_summary -file riffa_top_pcie_7x_v2_1_timing_summary_routed.rpt -pb riffa_top_pcie_7x_v2_1_timing_summary_routed.pb }
  write_checkpoint -force riffa_top_pcie_7x_v2_1_routed.dcp
  close_msg_db -file route_design.pb
} RESULT]
if {$rc} {
  step_failed route_design
  return -code error $RESULT
} else {
  end_step route_design
}

