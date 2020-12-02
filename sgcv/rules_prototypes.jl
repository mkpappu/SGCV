@structuredVariationalRule(:node_type => SwitchingGaussianControlledVariance,
                           :outbound_type => Message{Gaussian},
                           :inbound_types => (Nothing, Message{Gaussian}, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                           :name => SVBSwitchingGaussianControlledVarianceIn1MPPPP)

@structuredVariationalRule(:node_type => SwitchingGaussianControlledVariance,
                           :outbound_type => Message{Gaussian},
                           :inbound_types => (Message{Gaussian}, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                           :name => SVBSwitchingGaussianControlledVarianceMIn2PPPP)

@structuredVariationalRule(:node_type => SwitchingGaussianControlledVariance,
                           :outbound_type => Message{Function},
                           :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                           :name => SVBSwitchingGaussianControlledVariancePIn3PPP)

@structuredVariationalRule(:node_type => SwitchingGaussianControlledVariance,
                           :outbound_type => Message{Function},
                           :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                           :name => SVBSwitchingGaussianControlledVariancePPIn4PP)


@structuredVariationalRule(:node_type => SwitchingGaussianControlledVariance,
                           :outbound_type => Message{Function},
                           :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                           :name => SVBSwitchingGaussianControlledVariancePPPIn5P)


@structuredVariationalRule(:node_type => SwitchingGaussianControlledVariance,
                           :outbound_type =>  Message{Function},
                           :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                           :name => SVBSwitchingGaussianControlledVariancePPPIn5P)

@structuredVariationalRule(:node_type => SwitchingGaussianControlledVariance,
                           :outbound_type => Message{Categorical},
                           :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                           :name => SVBSwitchingGaussianControlledVariancePPPPIn6)

@structuredVariationalRule(:node_type => SwitchingGaussianControlledVariance,
                          :outbound_type => Message{Gaussian},
                          :inbound_types => (Nothing, Message{Function}, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                          :name => SVBSwitchingGaussianControlledVarianceIn1FPPPP)

@structuredVariationalRule(:node_type => SwitchingGaussianControlledVariance,
                          :outbound_type => Message{Gaussian},
                          :inbound_types => (Message{Function}, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                          :name => SVBSwitchingGaussianControlledVarianceFIn2PPPP)

@marginalRule(:node_type => SwitchingGaussianControlledVariance,
              :inbound_types => (Message{Gaussian}, Message{Gaussian}, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
              :name => MSwitchingGaussianControlledVariance)

@marginalRule(:node_type => SwitchingGaussianControlledVariance,
            :inbound_types => (Message{Gaussian}, Message{Function}, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
            :name => MSwitchingGaussianControlledVarianceGF)

@marginalRule(:node_type => SwitchingGaussianControlledVariance,
              :inbound_types => (Message{Function}, Message{Gaussian}, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
              :name => MSwitchingGaussianControlledVarianceFG)

@structuredVariationalRule(:node_type     => GaussianMeanPrecision,
                          :outbound_type => Message{GaussianMeanVariance},
                          :inbound_types => (Message{Function}, Nothing, ProbabilityDistribution),
                          :name          => SVBGaussianMeanPrecisionMFND)

@marginalRule(:node_type => GaussianMeanPrecision,
            :inbound_types => (Message{Function}, Message{Gaussian}, ProbabilityDistribution),
            :name => MGaussianMeanPrecisionFGD)
